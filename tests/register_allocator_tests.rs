//! Differential tests for the register allocator.
//!
//! Each test builds pseudo-random IR programs from a deterministic spec, evaluates that
//! spec directly in Rust, then compiles the IR, runs the generated machine code and
//! compares the two. This is what actually exercises spilling, reloads, block arguments,
//! loop back edges and values held live across calls -- the rest of the suite barely
//! reaches that code.
//!
//! `n_carried` is the number of values carried around the loop as block arguments, which
//! is the main pressure knob: on x86-64 Linux only 5 GPRs are allocatable, so a value
//! held live across the whole function plus several loop-carried values is enough to
//! force spilling.

use std::mem;

use dgbir::{
    compiler::compile,
    ir::{const_u32, const_u64, CompareType, DataType, IRBlockHandle, IRContext, IRFunction, InputSlot},
};

#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Shl(u32),
    Shr(u32),
    Not,
}

impl Op {
    fn emit(&self, block: &mut IRBlockHandle, a: InputSlot, b: InputSlot) -> InputSlot {
        match *self {
            Op::Add => block.add(DataType::U64, a, b).val(),
            Op::Sub => block.subtract(DataType::U64, a, b).val(),
            Op::And => block.and(DataType::U64, a, b).val(),
            Op::Or => block.or(DataType::U64, a, b).val(),
            Op::Xor => block.xor(DataType::U64, a, b).val(),
            Op::Shl(s) => block.left_shift(DataType::U64, a, const_u32(s)).val(),
            Op::Shr(s) => block.right_shift(DataType::U64, a, const_u32(s)).val(),
            Op::Not => block.not(DataType::U64, a).val(),
        }
    }

    fn eval(&self, a: u64, b: u64) -> u64 {
        match *self {
            Op::Add => a.wrapping_add(b),
            Op::Sub => a.wrapping_sub(b),
            Op::And => a & b,
            Op::Or => a | b,
            Op::Xor => a ^ b,
            Op::Shl(s) => a << s,
            Op::Shr(s) => a >> s,
            Op::Not => !a,
        }
    }
}

/// One SSA definition: `dst = op(vals[src_a], vals[src_b])`.
#[derive(Clone, Copy, Debug)]
struct Step {
    op: Op,
    src_a: usize,
    src_b: usize,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
}

/// The external function called from generated programs, to exercise saving live
/// values held in volatile registers across a call.
extern "C" fn mix64(x: u64) -> u64 {
    x.rotate_left(17) ^ 0x9E37_79B9_7F4A_7C15
}

/// A generated program: a straight-line prologue, then a loop body, then an epilogue
/// that stores the carried values.
struct Spec {
    /// How many values are carried around the loop as block arguments.
    n_carried: usize,
    prologue: Vec<Step>,
    /// Index into the prologue value list that gets passed through `mix64`.
    call_arg: usize,
    /// Which prologue values are carried into the loop.
    carried: Vec<usize>,
    body: Vec<Step>,
    /// Which body values are carried back around the loop.
    carry_back: Vec<usize>,
    loop_trips: u64,
}

impl Spec {
    fn generate(seed: u64, n_carried: usize, prologue_len: usize, body_len: usize, loop_trips: u64) -> Spec {
        let mut rng = Rng(seed | 1);
        let mut mk_steps = |len: usize, base: usize, rng: &mut Rng| -> Vec<Step> {
            (0..len)
                .map(|i| {
                    let n_available = base + i;
                    let op = match rng.below(8) {
                        0 => Op::Add,
                        1 => Op::Sub,
                        2 => Op::And,
                        3 => Op::Or,
                        4 => Op::Xor,
                        5 => Op::Shl(1 + rng.below(31) as u32),
                        6 => Op::Shr(1 + rng.below(31) as u32),
                        _ => Op::Not,
                    };
                    Step {
                        op,
                        src_a: rng.below(n_available),
                        src_b: rng.below(n_available),
                    }
                })
                .collect()
        };

        // Prologue starts with 2 block inputs available.
        let prologue = mk_steps(prologue_len, 2, &mut rng);
        let n_prologue = 2 + prologue_len;
        let call_arg = rng.below(n_prologue);
        // +1 for the call result
        let n_prologue = n_prologue + 1;
        let carried = (0..n_carried).map(|_| rng.below(n_prologue)).collect::<Vec<_>>();

        // Loop body starts with (counter + carried) block inputs available.
        let n_inputs = 1 + n_carried;
        let body = mk_steps(body_len, n_inputs, &mut rng);
        let n_body = n_inputs + body_len;
        let carry_back = (0..n_carried).map(|_| rng.below(n_body)).collect::<Vec<_>>();

        Spec {
            n_carried,
            prologue,
            call_arg,
            carried,
            body,
            carry_back,
            loop_trips,
        }
    }

    /// Reference implementation, evaluated directly in Rust.
    fn eval(&self, x0: u64, x1: u64) -> Vec<u64> {
        let mut vals = vec![x0, x1];
        for step in &self.prologue {
            let v = step.op.eval(vals[step.src_a], vals[step.src_b]);
            vals.push(v);
        }
        vals.push(mix64(vals[self.call_arg]));

        let mut counter = 0u64;
        let mut carried = self.carried.iter().map(|i| vals[*i]).collect::<Vec<_>>();

        loop {
            let mut vals = vec![counter];
            vals.extend_from_slice(&carried);
            for step in &self.body {
                let v = step.op.eval(vals[step.src_a], vals[step.src_b]);
                vals.push(v);
            }
            let next = self.carry_back.iter().map(|i| vals[*i]).collect::<Vec<_>>();
            counter += 1;
            if counter < self.loop_trips {
                carried = next;
            } else {
                return next;
            }
        }
    }

    fn build(&self) -> IRFunction {
        let func = IRFunction::new(IRContext::new());

        let mut entry = func.new_block(vec![DataType::Ptr, DataType::U64, DataType::U64]);
        // Referenced from the exit block; block 0 dominates everything, so this value stays
        // live across the whole function.
        let result_ptr = entry.input(0);
        let mut vals = vec![entry.input(1), entry.input(2)];
        for step in &self.prologue {
            let v = step.op.emit(&mut entry, vals[step.src_a], vals[step.src_b]);
            vals.push(v);
        }
        let call_result = entry
            .call_function(dgbir::ir::const_ptr(mix64 as usize), Some(DataType::U64), vec![vals[self.call_arg]])
            .val();
        vals.push(call_result);

        // loop(counter, carried...)
        let mut loop_inputs = vec![DataType::U64];
        loop_inputs.extend(std::iter::repeat(DataType::U64).take(self.n_carried));
        let mut loop_block = func.new_block(loop_inputs);
        let mut exit = func.new_block(std::iter::repeat(DataType::U64).take(self.n_carried).collect());

        let mut entry_args = vec![const_u64(0)];
        entry_args.extend(self.carried.iter().map(|i| vals[*i]));
        entry.jump(loop_block.call(entry_args));

        let mut body_vals = (0..=self.n_carried).map(|i| loop_block.input(i)).collect::<Vec<_>>();
        for step in &self.body {
            let v = step
                .op
                .emit(&mut loop_block, body_vals[step.src_a], body_vals[step.src_b]);
            body_vals.push(v);
        }
        let next = self.carry_back.iter().map(|i| body_vals[*i]).collect::<Vec<_>>();
        let counter2 = loop_block.add(DataType::U64, body_vals[0], const_u64(1)).val();
        let again = loop_block
            .compare(DataType::U64, counter2, CompareType::LessThan, const_u64(self.loop_trips))
            .val();

        let mut back_args = vec![counter2];
        back_args.extend(next.iter().cloned());
        let back = loop_block.call(back_args);
        let out = exit.call(next);
        loop_block.branch(again, back, out);

        for i in 0..self.n_carried {
            let v = exit.input(i);
            exit.write_ptr(DataType::U64, result_ptr, i * 8, v);
        }
        exit.ret(None);

        func
    }
}

/// Compiles and runs one generated program and compares it against the Rust reference.
/// Panics with the failing shape so a regression names itself.
fn check_case(label: &str, spec: &Spec, x0: u64, x1: u64) {
    let expected = spec.eval(x0, x1);

    let func = spec.build();
    func.validate();
    let compiled = compile(&func);
    let f: extern "C" fn(usize, u64, u64) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    let actual = vec![0u64; spec.n_carried];
    f(actual.as_ptr() as usize, x0, x1);

    assert_eq!(actual, expected, "{label} (x0={x0:#x}, x1={x1:#x})");
}

/// Runs every seed and input combination for a set of shapes.
fn check_shapes(shapes: &[(usize, usize, usize, u64)]) {
    for (shape_i, (n_carried, prologue_len, body_len, trips)) in shapes.iter().enumerate() {
        for seed in 1..=5u64 {
            let spec = Spec::generate(
                seed.wrapping_mul(0x1234_5678).wrapping_add(shape_i as u64),
                *n_carried,
                *prologue_len,
                *body_len,
                *trips,
            );
            for (x0, x1) in [(0u64, 0u64), (1, 2), (0xDEAD_BEEF_CAFE_BABE, 0x0123_4567_89AB_CDEF)] {
                let label =
                    format!("carried={n_carried} prologue={prologue_len} body={body_len} trips={trips} seed={seed}");
                check_case(&label, &spec, x0, x1);
            }
        }
    }
}

/// Exercises GPR and SIMD values live at the same time, across a call and a loop.
fn check_mixed_case(trips: u64, width: usize, x: u64, f0: f32) {
    let expected = {
        let mut ints: Vec<u64> = (0..width).map(|i| x.wrapping_add(i as u64)).collect();
        let mut flts: Vec<f32> = (0..width).map(|i| f0 + i as f32).collect();
        let mut acc_i = 0u64;
        let mut acc_f = 0.0f32;
        for i in 0..width {
            acc_i = acc_i.wrapping_add(ints[i]);
            acc_f += flts[i];
        }
        acc_i = mix64(acc_i);
        let mut counter = 0u64;
        loop {
            acc_i = acc_i.wrapping_add(counter).wrapping_add(1);
            acc_f += 1.0;
            counter += 1;
            if counter >= trips {
                break;
            }
        }
        ints.clear();
        flts.clear();
        (acc_i, acc_f)
    };

    let actual = {
        let func = IRFunction::new(IRContext::new());
        let mut entry = func.new_block(vec![DataType::Ptr, DataType::U64, DataType::F32]);
        let ptr = entry.input(0);
        let x_in = entry.input(1);
        let f_in = entry.input(2);

        // Build `width` int and float values, all kept live simultaneously.
        let ints: Vec<InputSlot> = (0..width)
            .map(|i| entry.add(DataType::U64, x_in, const_u64(i as u64)).val())
            .collect();
        let flts: Vec<InputSlot> = (0..width)
            .map(|i| entry.add(DataType::F32, f_in, dgbir::ir::const_f32(i as f32)).val())
            .collect();

        let mut acc_i = const_u64(0);
        let mut acc_f = dgbir::ir::const_f32(0.0);
        for i in 0..width {
            acc_i = entry.add(DataType::U64, acc_i, ints[i]).val();
            acc_f = entry.add(DataType::F32, acc_f, flts[i]).val();
        }

        // acc_f is live across this call, so it must be preserved.
        let acc_i = entry
            .call_function(dgbir::ir::const_ptr(mix64 as usize), Some(DataType::U64), vec![acc_i])
            .val();

        let mut lp = func.new_block(vec![DataType::U64, DataType::U64, DataType::F32]);
        let mut exit = func.new_block(vec![DataType::U64, DataType::F32]);
        entry.jump(lp.call(vec![const_u64(0), acc_i, acc_f]));

        let counter = lp.input(0);
        let ai = lp.add(DataType::U64, lp.input(1), counter).val();
        let ai = lp.add(DataType::U64, ai, const_u64(1)).val();
        let af = lp.add(DataType::F32, lp.input(2), dgbir::ir::const_f32(1.0)).val();
        let counter2 = lp.add(DataType::U64, counter, const_u64(1)).val();
        let again = lp
            .compare(DataType::U64, counter2, CompareType::LessThan, const_u64(trips))
            .val();
        let back = lp.call(vec![counter2, ai, af]);
        let out = exit.call(vec![ai, af]);
        lp.branch(again, back, out);

        let e0 = exit.input(0);
        let e1 = exit.input(1);
        exit.write_ptr(DataType::U64, ptr, 0, e0);
        exit.write_ptr(DataType::F32, ptr, 8, e1);
        exit.ret(None);

        func.validate();
        let compiled = compile(&func);
        let g: extern "C" fn(usize, u64, f32) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
        #[repr(C)]
        struct Out {
            i: u64,
            f: f32,
        }
        let out = Out { i: 0, f: 0.0 };
        g(&out as *const Out as usize, x, f0);
        (out.i, out.f)
    };

    assert_eq!(actual, expected, "mixed(trips={trips}, width={width}, x={x:#x}, f0={f0})");
}

/// One loop-carried value. The original graph-colouring allocator handled these.
#[test]
fn random_programs_one_carried_value() {
    check_shapes(&[(1, 2, 2, 2), (1, 8, 6, 3), (1, 20, 10, 4), (1, 60, 10, 2)]);
}

/// Two loop-carried values. Together with the result pointer held live across the whole
/// function this exceeds the 5 allocatable GPRs, so it needs accurate liveness across the
/// loop back edge. The original allocator could not allocate these at all.
#[test]
fn random_programs_two_carried_values() {
    check_shapes(&[(2, 4, 4, 3), (2, 12, 8, 5), (2, 24, 16, 4), (2, 40, 24, 3)]);
}

/// Three loop-carried values: the highest pressure the IR can express here.
#[test]
fn random_programs_three_carried_values() {
    check_shapes(&[(3, 8, 8, 3), (3, 20, 12, 4), (3, 40, 30, 5)]);
}

/// Integer and float values live simultaneously exercises both register classes, and the
/// float accumulator is live across a call so it must be preserved by the caller.
#[test]
fn mixed_int_and_float_across_call_and_loop() {
    for (trips, width, x, f0) in [
        (2u64, 3usize, 7u64, 1.5f32),
        (4, 6, 0xDEAD_BEEF, 0.25),
        (3, 10, 1, -2.5),
    ] {
        check_mixed_case(trips, width, x, f0);
    }
}
