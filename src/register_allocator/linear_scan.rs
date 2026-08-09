//! A linear-scan register allocator.
//!
//! DISCLAIMER: This file is LLM written, while the rest of the project is largely handwritten.
//!
//! # Overview
//!
//! The allocator works on a *linearized* view of the function: blocks are laid out in block-index
//! order (which is also the order [`crate::compiler`] emits them in) and every instruction gets a
//! program point. Liveness is computed with a standard backward dataflow fixpoint over the CFG, and
//! each value is then summarised as a single live interval `[start, end]` covering every point at
//! which it is live.
//!
//! Two values interfere exactly when their intervals overlap, so the interference graph is an
//! interval graph. Greedy left-to-right coloring is optimal on interval graphs: if no more than `K`
//! intervals of a register class overlap at any point, the sweep never runs out of registers. That
//! turns register allocation into a pressure-reduction problem:
//!
//! 1. [`analyze`] builds the point numbering, liveness and live intervals. `O(n)` modulo the
//!    dataflow fixpoint.
//! 2. [`plan_spills`] sweeps the points once, maintaining the set of values that currently occupy a
//!    register. Whenever the pressure of a class exceeds the number of registers it evicts the
//!    value whose next use is furthest away (Belady's rule). Evicted values are stored to the stack
//!    right after their definition and reloaded before the uses that follow the eviction.
//! 3. [`apply_plan`] performs all of the IR mutation in one batch.
//! 4. The whole thing is repeated on the mutated function. The second round normally finds nothing
//!    left to spill and hands the intervals to [`assign_registers`].
//!
//! Because liveness is exact, values that are live across a jump automatically interfere with the
//! block inputs of the jump target (they are live at the point where those inputs are defined), so
//! the parallel move that the compiler emits at a jump can never clobber a live value.
//!
//! Correctness of the final assignment only depends on the (recomputed) liveness of the mutated
//! function, not on the spill heuristics: spill decisions can only make the generated code slower,
//! never wrong.

use std::collections::BTreeMap;

use itertools::Itertools;

use crate::{
    abi::get_registers,
    ir::{
        const_ptr, BlockReference, Constant, DataType, IRFunctionInternal, IndexedInstruction, InputSlot, Instruction,
        InstructionType, OutputSlot,
    },
};

use super::{Lifetimes, LiveRange, Register, RegisterAllocations, Value};

/// Sentinel for "no value id".
const NO_VALUE: u32 = u32::MAX;
/// Sentinel for "no program point".
const NO_POINT: u32 = u32::MAX;

const N_CLASSES: usize = 2;
const CLASS_GPR: u8 = 0;
const CLASS_SIMD: u8 = 1;

/// How many times we are willing to re-run analysis + spilling before giving up. Two rounds is the
/// normal case (one to spill, one to confirm); more than that means the spill heuristic and the
/// recomputed liveness disagreed, which can only happen for pathological control flow.
const MAX_ROUNDS: usize = 64;

fn class_of(tp: DataType) -> u8 {
    if Register::GPR(0).can_hold_datatype(tp) {
        CLASS_GPR
    } else if Register::SIMD(0).can_hold_datatype(tp) {
        CLASS_SIMD
    } else {
        panic!("No register class can hold values of type {:?}", tp)
    }
}

/// The allocatable registers, split per class.
struct RegisterFile {
    regs: [Vec<Register>; N_CLASSES],
}

impl RegisterFile {
    fn new() -> Self {
        let all = get_registers();
        let gpr = all.iter().filter(|r| r.is_gpr()).copied().collect_vec();
        let simd = all.iter().filter(|r| r.is_simd()).copied().collect_vec();
        assert!(gpr.len() <= 32 && simd.len() <= 32, "Too many allocatable registers per class");
        RegisterFile { regs: [gpr, simd] }
    }

    #[inline]
    fn capacity(&self, class: u8) -> usize {
        self.regs[class as usize].len()
    }
}

// ---------------------------------------------------------------------------------------------
// Bit sets
// ---------------------------------------------------------------------------------------------

/// A dense matrix of bits, used for the per-block liveness sets.
struct BitMatrix {
    words: usize,
    data: Vec<u64>,
}

impl BitMatrix {
    fn new(rows: usize, bits: usize) -> Self {
        let words = (bits + 63) / 64;
        BitMatrix {
            words,
            data: vec![0; rows * words],
        }
    }

    #[inline]
    fn row(&self, r: usize) -> &[u64] {
        &self.data[r * self.words..(r + 1) * self.words]
    }

    #[inline]
    fn set(&mut self, r: usize, bit: usize) {
        let w = self.words;
        self.data[r * w + bit / 64] |= 1u64 << (bit % 64);
    }

    #[inline]
    fn unset(&mut self, r: usize, bit: usize) {
        let w = self.words;
        self.data[r * w + bit / 64] &= !(1u64 << (bit % 64));
    }

    /// Calls `f` for every set bit in row `r`.
    fn for_each(&self, r: usize, mut f: impl FnMut(usize)) {
        let row = self.row(r);
        for (i, word) in row.iter().enumerate() {
            let mut w = *word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                f(i * 64 + bit);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Instruction helpers
// ---------------------------------------------------------------------------------------------

fn for_each_input_slot(instruction: &Instruction, mut f: impl FnMut(&InputSlot)) {
    match instruction {
        #[cfg(feature = "ir_comments")]
        Instruction::Comment(_) => {}
        Instruction::Instruction { inputs, .. } => inputs.iter().for_each(|s| f(s)),
        Instruction::Branch {
            cond,
            if_true,
            if_false,
        } => {
            f(cond);
            if_true.arguments.iter().for_each(|s| f(s));
            if_false.arguments.iter().for_each(|s| f(s));
        }
        Instruction::Jump { target } => target.arguments.iter().for_each(|s| f(s)),
        Instruction::Return { value } => value.iter().for_each(|s| f(s)),
    }
}

fn for_each_input_slot_mut(instruction: &mut Instruction, mut f: impl FnMut(&mut InputSlot)) {
    fn block_ref(r: &mut BlockReference, f: &mut impl FnMut(&mut InputSlot)) {
        r.arguments.iter_mut().for_each(|s| f(s));
    }
    match instruction {
        #[cfg(feature = "ir_comments")]
        Instruction::Comment(_) => {}
        Instruction::Instruction { inputs, .. } => inputs.iter_mut().for_each(|s| f(s)),
        Instruction::Branch {
            cond,
            if_true,
            if_false,
        } => {
            f(cond);
            block_ref(if_true, &mut f);
            block_ref(if_false, &mut f);
        }
        Instruction::Jump { target } => block_ref(target, &mut f),
        Instruction::Return { value } => value.iter_mut().for_each(|s| f(s)),
    }
}

#[inline]
fn resolve_slot(slot: &InputSlot, instr_out_base: &[u32], block_in_base: &[u32]) -> Option<u32> {
    match slot {
        InputSlot::InstructionOutput {
            instruction_index,
            output_index,
            ..
        } => {
            let base = *instr_out_base.get(*instruction_index)?;
            if base == NO_VALUE {
                None
            } else {
                Some(base + *output_index as u32)
            }
        }
        InputSlot::BlockInput {
            block_index,
            input_index,
            ..
        } => {
            let base = *block_in_base.get(*block_index)?;
            if base == NO_VALUE {
                None
            } else {
                Some(base + *input_index as u32)
            }
        }
        InputSlot::Constant(_) => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------------------------

/// Everything the allocator needs to know about a function, indexed by dense ids.
struct Analysis {
    n_blocks: usize,
    n_points: u32,
    /// First program point of each block. For empty blocks this is the point of whatever follows.
    block_start: Vec<u32>,
    /// Number of instructions in each block.
    block_len: Vec<u32>,
    /// Program point -> index into `func.instructions`.
    point_instr: Vec<u32>,
    /// Program point -> block index.
    point_block: Vec<u32>,
    /// Lowest program point reachable from a block (including itself). A use before this point can
    /// never be executed again once control reaches the block.
    floor: Vec<u32>,

    n_values: usize,
    values: Vec<Value>,
    class: Vec<u8>,
    /// Point of the defining instruction, or `NO_POINT` for block inputs (they are defined before
    /// the first instruction of their block).
    def_point: Vec<u32>,
    /// The point at which the value's register lifetime begins.
    start: Vec<u32>,
    end: Vec<u32>,
    /// Values produced by a reload; spilling these again would not make progress.
    unspillable: Vec<bool>,

    /// CSR: uses of value `v` are `use_points[use_start[v]..use_start[v] + use_len[v]]`.
    use_start: Vec<u32>,
    use_len: Vec<u32>,
    use_points: Vec<u32>,
    /// For each entry of `use_points`, the point of the last use of the same value in the same block.
    use_group_end: Vec<u32>,

    /// CSR: values used at point `p` are `point_use_ids[point_use_start[p]..point_use_start[p+1]]`.
    point_use_start: Vec<u32>,
    point_use_ids: Vec<u32>,
    /// First value id defined at a point, and how many. Block inputs are not included.
    point_def_base: Vec<u32>,
    point_def_count: Vec<u32>,

    instr_out_base: Vec<u32>,
    block_in_base: Vec<u32>,

    /// Live value ids sorted by interval start.
    order: Vec<u32>,
}

impl Analysis {
    #[inline]
    fn is_live(&self, v: u32) -> bool {
        self.use_len[v as usize] > 0
    }

    #[inline]
    fn uses(&self, v: u32) -> std::ops::Range<usize> {
        let s = self.use_start[v as usize] as usize;
        s..s + self.use_len[v as usize] as usize
    }
}

fn analyze(func: &IRFunctionInternal) -> Analysis {
    let n_blocks = func.blocks.len();
    let n_instrs = func.instructions.len();

    // --- Program points -------------------------------------------------------------------
    let mut block_start = vec![0u32; n_blocks];
    let mut block_len = vec![0u32; n_blocks];
    let mut n_points: u32 = 0;
    for b in 0..n_blocks {
        block_start[b] = n_points;
        block_len[b] = func.blocks[b].instructions.len() as u32;
        n_points += block_len[b];
    }
    let mut point_instr: Vec<u32> = Vec::with_capacity(n_points as usize);
    let mut point_block: Vec<u32> = Vec::with_capacity(n_points as usize);
    for b in 0..n_blocks {
        for &i in &func.blocks[b].instructions {
            point_instr.push(i as u32);
            point_block.push(b as u32);
        }
    }
    let mut instr_point = vec![NO_POINT; n_instrs];
    for (p, &i) in point_instr.iter().enumerate() {
        instr_point[i as usize] = p as u32;
    }

    // --- Successors -----------------------------------------------------------------------
    let mut succ: Vec<Vec<u32>> = vec![Vec::new(); n_blocks];
    for b in 0..n_blocks {
        for &i in &func.blocks[b].instructions {
            match &func.instructions[i].instruction {
                Instruction::Branch { if_true, if_false, .. } => {
                    succ[b].push(if_true.block_index as u32);
                    succ[b].push(if_false.block_index as u32);
                }
                Instruction::Jump { target } => succ[b].push(target.block_index as u32),
                _ => {}
            }
        }
        succ[b].sort_unstable();
        succ[b].dedup();
    }

    // --- Value numbering ------------------------------------------------------------------
    let mut block_in_base = vec![NO_VALUE; n_blocks];
    let mut instr_out_base = vec![NO_VALUE; n_instrs];
    let mut values: Vec<Value> = Vec::new();
    let mut class: Vec<u8> = Vec::new();
    let mut def_point: Vec<u32> = Vec::new();
    let mut unspillable: Vec<bool> = Vec::new();
    let mut point_def_base = vec![NO_VALUE; n_points as usize];
    let mut point_def_count = vec![0u32; n_points as usize];

    for b in 0..n_blocks {
        block_in_base[b] = values.len() as u32;
        for (input_index, tp) in func.blocks[b].inputs.iter().enumerate() {
            values.push(Value::BlockInput {
                block_index: b,
                input_index,
                data_type: *tp,
            });
            class.push(class_of(*tp));
            def_point.push(NO_POINT);
            unspillable.push(false);
        }
        for (i_in_block, &instr_index) in func.blocks[b].instructions.iter().enumerate() {
            let indexed = &func.instructions[instr_index];
            let (tp, outputs) = match &indexed.instruction {
                Instruction::Instruction { tp, outputs, .. } => (Some(tp), outputs),
                _ => continue,
            };
            if outputs.is_empty() {
                continue;
            }
            let is_reload = matches!(tp, Some(InstructionType::LoadFromStack));
            let p = block_start[b] + i_in_block as u32;
            point_def_base[p as usize] = values.len() as u32;
            point_def_count[p as usize] = outputs.len() as u32;
            instr_out_base[instr_index] = values.len() as u32;
            for (output_index, output) in outputs.iter().enumerate() {
                values.push(Value::InstructionOutput {
                    // Must match what the compiler uses to look allocations up.
                    block_index: indexed.block_index,
                    instruction_index: instr_index,
                    output_index,
                    data_type: output.tp,
                });
                class.push(class_of(output.tp));
                def_point.push(p);
                unspillable.push(is_reload);
            }
        }
    }
    let n_values = values.len();

    // --- Uses -----------------------------------------------------------------------------
    // Counting pass (an instruction using the same value twice is counted twice; the fill pass
    // deduplicates and `use_len` records the real length).
    let mut use_start = vec![0u32; n_values + 1];
    let mut point_use_start = vec![0u32; n_points as usize + 1];
    for p in 0..n_points as usize {
        let instruction = &func.instructions[point_instr[p] as usize].instruction;
        for_each_input_slot(instruction, |slot| {
            if let Some(id) = resolve_slot(slot, &instr_out_base, &block_in_base) {
                use_start[id as usize + 1] += 1;
                point_use_start[p + 1] += 1;
            }
        });
    }
    for i in 0..n_values {
        use_start[i + 1] += use_start[i];
    }
    for p in 0..n_points as usize {
        point_use_start[p + 1] += point_use_start[p];
    }
    let total_uses = use_start[n_values] as usize;
    let mut use_points = vec![0u32; total_uses];
    let mut use_len = vec![0u32; n_values];
    let mut point_use_ids = vec![NO_VALUE; point_use_start[n_points as usize] as usize];
    let mut point_fill = point_use_start.clone();
    for p in 0..n_points as usize {
        let instruction = &func.instructions[point_instr[p] as usize].instruction;
        for_each_input_slot(instruction, |slot| {
            if let Some(id) = resolve_slot(slot, &instr_out_base, &block_in_base) {
                let len = &mut use_len[id as usize];
                let base = use_start[id as usize];
                // Uses are visited in point order, so duplicates within an instruction are adjacent.
                if *len == 0 || use_points[(base + *len - 1) as usize] != p as u32 {
                    use_points[(base + *len) as usize] = p as u32;
                    *len += 1;
                    point_use_ids[point_fill[p] as usize] = id;
                    point_fill[p] += 1;
                }
            }
        });
    }
    // Compact the per-point use lists (deduplication may have left holes).
    {
        let mut out = 0usize;
        let mut new_start = vec![0u32; n_points as usize + 1];
        for p in 0..n_points as usize {
            new_start[p] = out as u32;
            for i in point_use_start[p]..point_fill[p] {
                point_use_ids[out] = point_use_ids[i as usize];
                out += 1;
            }
        }
        new_start[n_points as usize] = out as u32;
        point_use_ids.truncate(out);
        point_use_start = new_start;
    }

    // For every use, the point of the last use of the same value inside the same block.
    let mut use_group_end = vec![0u32; total_uses];
    for v in 0..n_values {
        let range = use_start[v] as usize..(use_start[v] + use_len[v]) as usize;
        let mut i = range.end;
        while i > range.start {
            let block = point_block[use_points[i - 1] as usize];
            let group_end = use_points[i - 1];
            while i > range.start && point_block[use_points[i - 1] as usize] == block {
                use_group_end[i - 1] = group_end;
                i -= 1;
            }
        }
    }

    // --- Liveness -------------------------------------------------------------------------
    let mut gen = BitMatrix::new(n_blocks, n_values);
    let mut def = BitMatrix::new(n_blocks, n_values);
    for b in 0..n_blocks {
        for &instr_index in func.blocks[b].instructions.iter().rev() {
            if let Instruction::Instruction { outputs, .. } = &func.instructions[instr_index].instruction {
                let base = instr_out_base[instr_index];
                if base != NO_VALUE {
                    for j in 0..outputs.len() {
                        gen.unset(b, base as usize + j);
                        def.set(b, base as usize + j);
                    }
                }
            }
            for_each_input_slot(&func.instructions[instr_index].instruction, |slot| {
                if let Some(id) = resolve_slot(slot, &instr_out_base, &block_in_base) {
                    gen.set(b, id as usize);
                }
            });
        }
        let base = block_in_base[b];
        for j in 0..func.blocks[b].inputs.len() {
            gen.unset(b, base as usize + j);
            def.set(b, base as usize + j);
        }
    }

    let words = gen.words;
    let mut live_in = BitMatrix::new(n_blocks, n_values);
    let mut live_out = BitMatrix::new(n_blocks, n_values);
    live_in.data.copy_from_slice(&gen.data);
    let mut scratch = vec![0u64; words];
    loop {
        let mut changed = false;
        for b in (0..n_blocks).rev() {
            scratch.iter_mut().for_each(|w| *w = 0);
            for &s in &succ[b] {
                let row = live_in.row(s as usize);
                for w in 0..words {
                    scratch[w] |= row[w];
                }
            }
            {
                let out_row_start = b * words;
                live_out.data[out_row_start..out_row_start + words].copy_from_slice(&scratch);
            }
            let gen_row_start = b * words;
            for w in 0..words {
                let value = gen.data[gen_row_start + w] | (scratch[w] & !def.data[gen_row_start + w]);
                if live_in.data[gen_row_start + w] != value {
                    live_in.data[gen_row_start + w] = value;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // --- Live intervals -------------------------------------------------------------------
    let mut start = vec![NO_POINT; n_values];
    let mut end = vec![0u32; n_values];
    for v in 0..n_values {
        if use_len[v] == 0 {
            continue;
        }
        let first_use = use_points[use_start[v] as usize];
        let last_use = use_points[(use_start[v] + use_len[v] - 1) as usize];
        let d = def_point[v];
        start[v] = if d == NO_POINT {
            first_use.min(block_start[block_of_value(&values[v])])
        } else {
            d.min(first_use)
        };
        end[v] = last_use.max(if d == NO_POINT { 0 } else { d });
    }
    for b in 0..n_blocks {
        if block_len[b] == 0 {
            continue;
        }
        let bs = block_start[b];
        let be = bs + block_len[b] - 1;
        live_in.for_each(b, |v| {
            if use_len[v] > 0 && bs < start[v] {
                start[v] = bs;
            }
        });
        live_out.for_each(b, |v| {
            if use_len[v] > 0 && be > end[v] {
                end[v] = be;
            }
        });
    }

    // --- Reachability floor ---------------------------------------------------------------
    let mut floor: Vec<u32> = block_start.clone();
    loop {
        let mut changed = false;
        for b in (0..n_blocks).rev() {
            let mut lowest = floor[b];
            for &s in &succ[b] {
                lowest = lowest.min(floor[s as usize]);
            }
            if lowest < floor[b] {
                floor[b] = lowest;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut order: Vec<u32> = (0..n_values as u32).filter(|v| use_len[*v as usize] > 0).collect();
    order.sort_unstable_by_key(|v| start[*v as usize]);

    Analysis {
        n_blocks,
        n_points,
        block_start,
        block_len,
        point_instr,
        point_block,
        floor,
        n_values,
        values,
        class,
        def_point,
        start,
        end,
        unspillable,
        use_start,
        use_len,
        use_points,
        use_group_end,
        point_use_start,
        point_use_ids,
        point_def_base,
        point_def_count,
        instr_out_base,
        block_in_base,
        order,
    }
}

fn block_of_value(v: &Value) -> usize {
    match v {
        Value::BlockInput { block_index, .. } => *block_index,
        Value::InstructionOutput { block_index, .. } => *block_index,
    }
}

// ---------------------------------------------------------------------------------------------
// Spill planning
// ---------------------------------------------------------------------------------------------

/// One entry of the simulated register file.
#[derive(Clone, Copy)]
struct Active {
    value: u32,
    end: u32,
    /// True when this entry is a reload of an already-spilled value rather than the value itself.
    reload: bool,
}

struct Plan {
    /// Values that get stored to the stack right after their definition.
    spilled: Vec<u32>,
    /// `(value, point of the first use served by this reload)`, in point order.
    reloads: Vec<(u32, u32)>,
}

impl Plan {
    fn is_empty(&self) -> bool {
        self.spilled.is_empty()
    }
}

struct Sweeper<'a> {
    a: &'a Analysis,
    file: &'a RegisterFile,
    /// Per value: index of the first use at a point >= the current point.
    cursor: Vec<u32>,
    /// Values that are spilled from their definition onwards.
    global: Vec<bool>,
    /// Values that have been evicted; uses after the eviction point are reloaded.
    evicted_at: Vec<u32>,
    active: Vec<Active>,
    pinned: Vec<u32>,
}

impl<'a> Sweeper<'a> {
    fn new(a: &'a Analysis, file: &'a RegisterFile) -> Self {
        Sweeper {
            a,
            file,
            cursor: vec![0; a.n_values],
            global: vec![false; a.n_values],
            evicted_at: vec![NO_POINT; a.n_values],
            active: Vec::new(),
            pinned: Vec::new(),
        }
    }

    /// Advances the per-value use cursor so that it points at the first use at a point `>= p`.
    #[inline]
    fn advance(&mut self, v: u32, p: u32) {
        let range = self.a.uses(v);
        let mut c = self.cursor[v as usize] as usize;
        if c < range.start {
            c = range.start;
        }
        while c < range.end && self.a.use_points[c] < p {
            c += 1;
        }
        self.cursor[v as usize] = c as u32;
    }

    #[inline]
    fn next_use_after(&self, v: u32, p: u32) -> u32 {
        let range = self.a.uses(v);
        let mut c = self.cursor[v as usize] as usize;
        if c < range.end && self.a.use_points[c] == p {
            c += 1;
        }
        if c < range.end {
            self.a.use_points[c]
        } else {
            NO_POINT
        }
    }

    /// True when the value has a use in `[floor, p]`, i.e. a use that might still be executed after
    /// control has reached `p`. Such a value must keep its register.
    #[inline]
    fn has_recent_use(&self, v: u32, p: u32, floor: u32) -> bool {
        let range = self.a.uses(v);
        let c = self.cursor[v as usize] as usize;
        if c < range.end && self.a.use_points[c] == p {
            return true;
        }
        c > range.start && self.a.use_points[c - 1] >= floor
    }

    #[inline]
    fn is_spilled_before(&self, v: u32, p: u32) -> bool {
        self.global[v as usize] || {
            let e = self.evicted_at[v as usize];
            e != NO_POINT && e < p
        }
    }

    fn count(&self, class: u8) -> usize {
        self.active
            .iter()
            .filter(|e| self.a.class[e.value as usize] == class)
            .count()
    }

    /// Runs one full sweep. Returns `Err(values)` when a point could not be brought under the
    /// register limit by evicting; those values have to be spilled everywhere and the sweep retried.
    fn run(&mut self, global: &[bool]) -> Result<Plan, Vec<u32>> {
        self.cursor.iter_mut().for_each(|c| *c = 0);
        self.evicted_at.iter_mut().for_each(|c| *c = NO_POINT);
        self.global.copy_from_slice(global);
        self.active.clear();

        let a = self.a;
        let mut reloads: Vec<(u32, u32)> = Vec::new();
        let mut order_cursor = 0usize;

        for p in 0..a.n_points {
            let block = a.point_block[p as usize] as usize;

            // 1. Retire values whose interval has ended.
            self.active.retain(|e| e.end >= p);

            // 2. Start the intervals that begin here.
            while order_cursor < a.order.len() && a.start[a.order[order_cursor] as usize] <= p {
                let v = a.order[order_cursor];
                order_cursor += 1;
                if !self.global[v as usize] {
                    self.active.push(Active {
                        value: v,
                        end: a.end[v as usize],
                        reload: false,
                    });
                }
            }

            // 3. Values written by this instruction always need a register, even when they are
            //    spilled immediately afterwards.
            self.pinned.clear();
            let def_base = a.point_def_base[p as usize];
            if def_base != NO_VALUE {
                for j in 0..a.point_def_count[p as usize] {
                    let v = def_base + j;
                    if !a.is_live(v) {
                        continue;
                    }
                    self.pinned.push(v);
                    if self.global[v as usize] {
                        // Spilled from birth: it only occupies a register for this instruction.
                        self.active.push(Active {
                            value: v,
                            end: p,
                            reload: false,
                        });
                    }
                }
            }

            // 4. Values read by this instruction need a register; spilled ones get a reload.
            let uses = a.point_use_start[p as usize] as usize..a.point_use_start[p as usize + 1] as usize;
            for &v in &a.point_use_ids[uses] {
                self.advance(v, p);
                self.pinned.push(v);
                if self.is_spilled_before(v, p) {
                    let has_reload = self.active.iter().any(|e| e.reload && e.value == v);
                    if !has_reload {
                        let group_end = a.use_group_end[self.cursor[v as usize] as usize];
                        self.active.push(Active {
                            value: v,
                            end: group_end,
                            reload: true,
                        });
                        reloads.push((v, p));
                    }
                }
            }

            // 5. Evict until every class fits.
            for class in 0..N_CLASSES as u8 {
                let capacity = self.file.capacity(class);
                while self.count(class) > capacity {
                    match self.pick_victim(class, p, a.floor[block]) {
                        Some(idx) => {
                            let entry = self.active[idx];
                            if !entry.reload {
                                self.evicted_at[entry.value as usize] = p;
                            }
                            self.active.swap_remove(idx);
                        }
                        None => {
                            let forced = self.pick_forced_global(class, p);
                            if forced.is_empty() {
                                panic!(
                                    "Register allocation failed: instruction at block {} index {} needs more \
                                     {} registers than the {} available",
                                    block,
                                    p - a.block_start[block],
                                    if class == CLASS_GPR { "GPR" } else { "SIMD" },
                                    capacity
                                );
                            }
                            return Err(forced);
                        }
                    }
                }
            }
        }

        let mut spilled = (0..a.n_values as u32)
            .filter(|v| self.global[*v as usize] || self.evicted_at[*v as usize] != NO_POINT)
            .collect_vec();
        spilled.sort_unstable();
        Ok(Plan { spilled, reloads })
    }

    /// Belady's rule: evict whatever is needed furthest in the future.
    fn pick_victim(&self, class: u8, p: u32, floor: u32) -> Option<usize> {
        let mut best: Option<(u32, usize)> = None;
        for (i, e) in self.active.iter().enumerate() {
            if self.a.class[e.value as usize] != class {
                continue;
            }
            if self.pinned.contains(&e.value) {
                continue;
            }
            let next = self.next_use_after(e.value, p);
            if !e.reload {
                // Spilling a value only helps if there is a later use to reload it at, and it is
                // only safe if no earlier use of it can be reached again from here.
                if next == NO_POINT || self.a.unspillable[e.value as usize] {
                    continue;
                }
                if self.has_recent_use(e.value, p, floor) {
                    continue;
                }
            } else if next == NO_POINT {
                // A reload that is no longer needed; drop it immediately.
                return Some(i);
            }
            if best.map(|(d, _)| next > d).unwrap_or(true) {
                best = Some((next, i));
            }
        }
        best.map(|(_, i)| i)
    }

    /// No value could be evicted at `p`. Pick values to spill everywhere instead (this is what
    /// loop-carried values need: their uses are at the top of the loop, before `p`).
    fn pick_forced_global(&self, class: u8, p: u32) -> Vec<u32> {
        let mut candidates = self
            .active
            .iter()
            .filter(|e| {
                self.a.class[e.value as usize] == class
                    && !e.reload
                    && !self.global[e.value as usize]
                    && !self.a.unspillable[e.value as usize]
                    && !self.pinned.contains(&e.value)
            })
            .map(|e| e.value)
            .collect_vec();
        if candidates.is_empty() {
            return candidates;
        }
        // Prefer values that are live for a long time but rarely used.
        candidates.sort_unstable_by_key(|v| {
            let span = self.a.end[*v as usize] - self.a.start[*v as usize];
            let uses = self.a.use_len[*v as usize];
            (std::cmp::Reverse(span / (uses + 1)), self.next_use_after(*v, p))
        });
        let needed = self.count(class) - self.file.capacity(class);
        candidates.truncate(needed.max(1));
        candidates
    }
}

fn plan_spills(a: &Analysis, file: &RegisterFile) -> Plan {
    let mut sweeper = Sweeper::new(a, file);
    let mut global = vec![false; a.n_values];
    loop {
        match sweeper.run(&global) {
            Ok(plan) => return plan,
            Err(forced) => {
                for v in forced {
                    global[v as usize] = true;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Applying a plan to the IR
// ---------------------------------------------------------------------------------------------

fn apply_plan(func: &mut IRFunctionInternal, a: &Analysis, plan: &Plan) {
    let mut insert_before: Vec<Vec<usize>> = vec![Vec::new(); a.n_points as usize];
    let mut insert_after: Vec<Vec<usize>> = vec![Vec::new(); a.n_points as usize];

    // Stack slot per spilled value, plus the store right after its definition.
    let mut stack_slot: BTreeMap<u32, usize> = BTreeMap::new();
    for &v in &plan.spilled {
        let value = a.values[v as usize];
        let tp = value.data_type();
        let location = func.new_stack_location(tp);
        stack_slot.insert(v, location);

        let block_index = block_of_value(&value);
        let index = func.instructions.len();
        func.instructions.push(IndexedInstruction {
            block_index,
            index,
            instruction: Instruction::Instruction {
                tp: InstructionType::SpillToStack,
                inputs: vec![
                    value.into_inputslot(),
                    const_ptr(location),
                    InputSlot::Constant(Constant::DataType(tp)),
                ],
                outputs: vec![],
            },
        });

        match a.def_point[v as usize] {
            // Block inputs are live before the first instruction of their block.
            NO_POINT => insert_before[a.block_start[block_index] as usize].push(index),
            def => insert_after[def as usize].push(index),
        }
    }

    // Reloads, and the rewrites that redirect uses to them.
    let mut groups: BTreeMap<u32, Vec<(u32, InputSlot)>> = BTreeMap::new();
    for &(v, first_use) in &plan.reloads {
        let value = a.values[v as usize];
        let tp = value.data_type();
        let location = stack_slot[&v];
        let block_index = a.point_block[first_use as usize] as usize;
        let index = func.instructions.len();
        func.instructions.push(IndexedInstruction {
            block_index,
            index,
            instruction: Instruction::Instruction {
                tp: InstructionType::LoadFromStack,
                inputs: vec![const_ptr(location)],
                outputs: vec![OutputSlot { tp }],
            },
        });
        insert_before[first_use as usize].push(index);
        groups.entry(v).or_default().push((
            first_use,
            InputSlot::InstructionOutput {
                instruction_index: index,
                output_index: 0,
                tp,
            },
        ));
    }

    // Each use is served by the last reload of the same value that precedes it in the same block.
    let mut rewrites: BTreeMap<usize, Vec<(u32, InputSlot)>> = BTreeMap::new();
    for (&v, group) in &groups {
        for use_index in a.uses(v) {
            let p = a.use_points[use_index];
            let block = a.point_block[p as usize];
            let reload = group
                .iter()
                .rev()
                .find(|(start, _)| *start <= p && a.point_block[*start as usize] == block);
            if let Some((_, slot)) = reload {
                rewrites
                    .entry(a.point_instr[p as usize] as usize)
                    .or_default()
                    .push((v, *slot));
            }
        }
    }
    for (instr_index, replacements) in rewrites {
        let instr_out_base = &a.instr_out_base;
        let block_in_base = &a.block_in_base;
        for_each_input_slot_mut(&mut func.instructions[instr_index].instruction, |slot| {
            if let Some(id) = resolve_slot(slot, instr_out_base, block_in_base) {
                if let Some((_, new_slot)) = replacements.iter().find(|(v, _)| *v == id) {
                    *slot = *new_slot;
                }
            }
        });
    }

    // Splice everything into the blocks in one pass.
    for b in 0..a.n_blocks {
        if a.block_len[b] == 0 {
            continue;
        }
        let base = a.block_start[b] as usize;
        let old = std::mem::take(&mut func.blocks[b].instructions);
        let extra: usize = (0..a.block_len[b] as usize)
            .map(|i| insert_before[base + i].len() + insert_after[base + i].len())
            .sum();
        if extra == 0 {
            func.blocks[b].instructions = old;
            continue;
        }
        let mut new_list = Vec::with_capacity(old.len() + extra);
        for (i, instr) in old.into_iter().enumerate() {
            new_list.extend_from_slice(&insert_before[base + i]);
            new_list.push(instr);
            new_list.extend_from_slice(&insert_after[base + i]);
        }
        func.blocks[b].instructions = new_list;
    }
}

// ---------------------------------------------------------------------------------------------
// Register assignment
// ---------------------------------------------------------------------------------------------

/// Greedy interval-graph coloring. Returns `None` if a register class ran out, which can only
/// happen when the pressure was not reduced far enough.
fn assign_registers(a: &Analysis, file: &RegisterFile) -> Option<Vec<u8>> {
    const NO_REG: u8 = u8::MAX;
    let mut assignment = vec![NO_REG; a.n_values];
    let mut free: [u32; N_CLASSES] = [0; N_CLASSES];
    for c in 0..N_CLASSES {
        free[c] = if file.regs[c].len() == 32 {
            u32::MAX
        } else {
            (1u32 << file.regs[c].len()) - 1
        };
    }
    // (end, register) pairs, per class. Never larger than the number of registers in the class.
    let mut active: [Vec<(u32, u8)>; N_CLASSES] = [Vec::new(), Vec::new()];

    for &v in &a.order {
        let class = a.class[v as usize] as usize;
        let start = a.start[v as usize];
        for c in 0..N_CLASSES {
            active[c].retain(|(end, reg)| {
                if *end < start {
                    free[c] |= 1 << *reg;
                    false
                } else {
                    true
                }
            });
        }
        if free[class] == 0 {
            return None;
        }
        let reg = free[class].trailing_zeros() as u8;
        free[class] &= !(1u32 << reg);
        assignment[v as usize] = reg;
        active[class].push((a.end[v as usize], reg));
    }
    Some(assignment)
}

// ---------------------------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------------------------

pub(super) fn regalloc(func: &mut IRFunctionInternal) -> RegisterAllocations {
    let file = RegisterFile::new();

    for round in 0.. {
        let a = analyze(func);
        let plan = plan_spills(&a, &file);
        if plan.is_empty() {
            if let Some(assignment) = assign_registers(&a, &file) {
                return finish(func, &a, &file, &assignment);
            }
        }
        assert!(round < MAX_ROUNDS, "Register allocation did not converge after {} rounds", MAX_ROUNDS);
        assert!(!plan.is_empty(), "Register allocation ran out of registers but found nothing to spill");
        apply_plan(func, &a, &plan);
    }
    unreachable!()
}

fn finish(func: &mut IRFunctionInternal, a: &Analysis, file: &RegisterFile, assignment: &[u8]) -> RegisterAllocations {
    let mut allocations: BTreeMap<Value, Register> = BTreeMap::new();
    let mut used: [u32; N_CLASSES] = [0; N_CLASSES];
    for &v in &a.order {
        let class = a.class[v as usize] as usize;
        let reg = assignment[v as usize];
        used[class] |= 1 << reg;
        allocations.insert(a.values[v as usize], file.regs[class][reg as usize]);
    }

    let callee_saved = (0..N_CLASSES)
        .flat_map(|c| {
            (0..file.regs[c].len())
                .filter(move |r| used[c] & (1 << r) != 0)
                .map(move |r| file.regs[c][r])
        })
        .filter(|reg| !reg.is_volatile())
        .map(|reg| (reg, func.new_sized_stack_location(reg.size())))
        .collect_vec();

    RegisterAllocations {
        allocations,
        callee_saved,
        lifetimes: build_lifetimes(a),
    }
}

fn build_lifetimes(a: &Analysis) -> Lifetimes {
    let live_ranges = a
        .order
        .iter()
        .map(|&v| LiveRange {
            start: a.start[v as usize],
            end: a.end[v as usize],
            def: a.def_point[v as usize],
            value: a.values[v as usize],
        })
        .collect_vec();

    // `a.order` is sorted by interval start, which is what `get_active_at_index` binary searches on.
    debug_assert!(live_ranges.windows(2).all(|w| w[0].start <= w[1].start));

    Lifetimes {
        live_ranges,
        block_starts: a.block_start.clone(),
    }
}
