use std::mem::{self, offset_of};

use dgbir::external_fn;
use dgbir::{
    abi::get_registers,
    compiler::compile,
    disassembler::disassemble_function,
    ir::{
        const_f32, const_s32, const_u32, const_u64, CompareType, Constant, DataType, IRBlockHandle, IRContext,
        IRFunction, InputSlot, LaneClass, MultiplyType, PackType, VectorHalf, VectorType,
    },
    ir_interpreter::interpret_func,
};

fn validate<T: std::fmt::Display + std::fmt::Debug + std::cmp::PartialEq>(results: &[T], expected: &[T]) {
    assert_eq!(
        results.len(),
        expected.len(),
        "Results length mismatch: expected {}, got {}",
        expected.len(),
        results.len()
    );
    for (i, v) in expected.iter().enumerate() {
        assert_eq!(results[i], *v, "Validation failed at index {}: expected {}, got {}", i, v, results[i]);
    }
}

#[test]
#[should_panic(expected = "Unclosed block")]
fn unclosed_block() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::U32]);
    block.add(DataType::U32, block.input(0), const_u32(1));
    // No return statement, block is unclosed
    println!("{}", func);
    println!("Compiling...");
    compile(&func);
}

#[test]
#[should_panic(
    expected = "Instruction 'v3 : U32 = Add(v2, U32(1), AddType(Wrapping))' references a value from block b1 which may not be initialized."
)]
fn use_value_from_non_dominator_block() {
    let context = IRContext::new();
    let func = IRFunction::new(context);

    let mut block_a = func.new_block(vec![DataType::Bool]);
    let one = block_a.add(DataType::U32, const_u32(1), const_u32(0));

    let mut block_b = func.new_block(vec![]);

    let mut block_c = func.new_block(vec![]);
    block_a.branch(block_a.input(0), block_b.call(vec![]), block_c.call(vec![]));

    let two = block_b.add(DataType::U32, one.val(), const_u32(1));

    let three = block_c.add(DataType::U32, two.val(), const_u32(1));

    let mut block_d = func.new_block(vec![DataType::U32]);
    block_b.jump(block_d.call(vec![two.val()]));
    block_c.jump(block_d.call(vec![three.val()]));
    block_d.ret(Some(block_d.input(0)));
    compile(&func);
}

#[test]
fn compiler_identityfunc() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::U32]);
    block.ret(Some(block.input(0)));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&func);
    let f: extern "C" fn(u32) -> u32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    assert_eq!(f(0), 0);
    assert_eq!(f(1), 1);
    assert_eq!(f(2), 2);
    assert_eq!(f(10000), 10000);
}

#[test]
fn compiler_addone() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::U32]);
    let add_result = block.add(DataType::U32, block.input(0), const_u32(1));
    block.ret(Some(add_result.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&func);
    let f: extern "C" fn(u32) -> u32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    assert_eq!(f(0), 1);
    assert_eq!(f(1), 2);
    assert_eq!(f(2), 3);
    assert_eq!(f(10000), 10001);
}

#[test]
fn compiler_identityfunc_f32() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::F32]);
    block.ret(Some(block.input(0)));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&func);
    let f: extern "C" fn(f32) -> f32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    assert_eq!(f(0.0), 0.0);
    assert_eq!(f(1.0), 1.0);
    assert_eq!(f(2.0), 2.0);
    assert_eq!(f(10000.0), 10000.0);
}

#[test]
fn compiler_addone_f32() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::F32]);
    let add_result = block.add(DataType::F32, block.input(0), const_f32(1.0));
    block.ret(Some(add_result.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&func);
    let f: extern "C" fn(f32) -> f32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble_function(&compiled));
    assert_eq!(f(0.0), 1.0);
    assert_eq!(f(1.0), 2.0);
    assert_eq!(f(2.0), 3.0);
    assert_eq!(f(10000.0), 10001.0);
}

#[test]
fn compiler_add_f32_to_self() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::F32]);
    let add_result = block.add(DataType::F32, block.input(0), block.input(0));
    block.ret(Some(add_result.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&func);
    let f: extern "C" fn(f32) -> f32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble_function(&compiled));
    assert_eq!(f(0.0), 0.0);
    assert_eq!(f(1.0), 2.0);
    assert_eq!(f(2.0), 4.0);
    assert_eq!(f(10000.0), 20000.0);
}

#[test]
fn constant_shifts_8() {
    let results: Vec<u64> = vec![0; 16];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::U8]);
    let result_ptr = block.input(0);
    let input = block.input(1);
    let mut index = 0;
    for tp in vec![DataType::U8, DataType::S8] {
        for const_shift_amount in vec![0, 1, 6, 8] {
            let left_result = block.left_shift(tp, input, const_u32(const_shift_amount));
            let right_result = block.right_shift(tp, input, const_u32(const_shift_amount));

            // Write the entire 64 bit register even though we only did a 32 bit shift to ensure
            // the behavior is consistent
            block.write_ptr(DataType::U64, result_ptr, index * size_of::<u64>(), left_result.val());
            index += 1;
            block.write_ptr(DataType::U64, result_ptr, index * size_of::<u64>(), right_result.val());
            index += 1;
        }
    }

    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, u8) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    // println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize, 2);
    println!("Shift 2: Results:");
    for (i, r) in results.iter().enumerate() {
        println!("{}: 0x{:08X}, ", i, r);
    }
    validate(
        &results,
        &[
            // U8
            2,   // << 0
            2,   // >> 0
            4,   // << 1
            1,   // >> 1
            128, // << 6
            0,   // >> 6
            2,   // << 8
            2,   // >> 8
            // S8
            2,   // << 0
            2,   // >> 0
            4,   // << 1
            1,   // >> 1
            128, // << 6
            0,   // >> 6
            2,   // << 8
            2,   // >> 8
        ],
    );

    f(results.as_ptr() as usize, 0xFF);
    println!("Shift 0xFF: Results:");

    for (i, r) in results.iter().enumerate() {
        println!("{}: 0x{:02X}, ", i, r);
    }
    validate(
        &results,
        &[
            // U8
            0xFF, // << 0
            0xFF, // >> 0
            0xFE, // << 1
            0x7F, // >> 1
            0xC0, // << 6
            0x03, // >> 6
            0xFF, // << 8
            0xFF, // >> 8
            // S8
            0xFF, // << 0
            0xFF, // >> 0
            0xFE, // << 1
            0xFF, // >> 1
            0xC0, // << 6
            0xFF, // >> 6
            0xFF, // << 8
            0xFF, // >> 8
        ],
    );
}

#[test]
fn constant_shifts_16() {
    let results: Vec<u64> = vec![0; 16];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::U16]);
    let result_ptr = block.input(0);
    let input = block.input(1);
    let mut index = 0;
    for tp in vec![DataType::U16, DataType::S16] {
        for const_shift_amount in vec![0, 1, 6, 32] {
            let left_result = block.left_shift(tp, input, const_u32(const_shift_amount));
            let right_result = block.right_shift(tp, input, const_u32(const_shift_amount));

            // Write the entire 64 bit register even though we only did a 32 bit shift to ensure
            // the behavior is consistent
            block.write_ptr(DataType::U64, result_ptr, index * size_of::<u64>(), left_result.val());
            index += 1;
            block.write_ptr(DataType::U64, result_ptr, index * size_of::<u64>(), right_result.val());
            index += 1;
        }
    }

    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, u16) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize, 2);
    println!("Shift 2: Results:");
    for (i, r) in results.iter().enumerate() {
        println!("{}: 0x{:08X}, ", i, r);
    }
    validate(
        &results,
        &[
            // U16
            2,   // << 0
            2,   // >> 0
            4,   // << 1
            1,   // >> 1
            128, // << 6
            0,   // >> 6
            2,   // << 16
            2,   // >> 16
            // S16
            2,   // << 0
            2,   // >> 0
            4,   // << 1
            1,   // >> 1
            128, // << 6
            0,   // >> 6
            2,   // << 16
            2,   // >> 16
        ],
    );

    f(results.as_ptr() as usize, 0xFFFF);
    println!("Shift 0xFFFF: Results:");

    for (i, r) in results.iter().enumerate() {
        println!("{}: 0x{:04X}, ", i, r);
    }
    validate(
        &results,
        &[
            // U16
            0xFFFF, // << 0
            0xFFFF, // >> 0
            0xFFFE, // << 1
            0x7FFF, // >> 1
            0xFFC0, // << 6
            0x03FF, // >> 6
            0xFFFF, // << 16
            0xFFFF, // >> 16
            // S16
            0xFFFF, // << 0
            0xFFFF, // >> 0
            0xFFFE, // << 1
            0xFFFF, // >> 1
            0xFFC0, // << 6
            0xFFFF, // >> 6
            0xFFFF, // << 16
            0xFFFF, // >> 16
        ],
    );
}

#[test]
fn constant_shifts_32() {
    let results: Vec<u64> = vec![0; 16];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::U32]);
    let result_ptr = block.input(0);
    let input = block.input(1);
    let mut index = 0;
    for tp in vec![DataType::U32, DataType::S32] {
        for const_shift_amount in vec![0, 1, 6, 32] {
            let left_result = block.left_shift(tp, input, const_u32(const_shift_amount));
            let right_result = block.right_shift(tp, input, const_u32(const_shift_amount));

            // Write the entire 64 bit register even though we only did a 32 bit shift to ensure
            // the behavior is consistent
            block.write_ptr(DataType::U64, result_ptr, index * size_of::<u64>(), left_result.val());
            index += 1;
            block.write_ptr(DataType::U64, result_ptr, index * size_of::<u64>(), right_result.val());
            index += 1;
        }
    }

    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, u32) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    f(results.as_ptr() as usize, 2);
    println!("Shift 2: Results:");
    for (i, r) in results.iter().enumerate() {
        println!("{}: 0x{:08X}, ", i, r);
    }
    validate(
        &results,
        &[
            // U32
            2,   // << 0
            2,   // >> 0
            4,   // << 1
            1,   // >> 1
            128, // << 6
            0,   // >> 6
            2,   // << 32
            2,   // >> 32
            // S32
            2,   // << 0
            2,   // >> 0
            4,   // << 1
            1,   // >> 1
            128, // << 6
            0,   // >> 6
            2,   // << 32
            2,   // >> 32
        ],
    );

    f(results.as_ptr() as usize, 0xFFFF_FFFF);
    println!("Shift 0xFFFF_FFFF: Results:");

    for (i, r) in results.iter().enumerate() {
        println!("{}: 0x{:08X}, ", i, r);
    }
    validate(
        &results,
        &[
            // U32
            0xFFFF_FFFF, // << 0
            0xFFFF_FFFF, // >> 0
            0xFFFF_FFFE, // << 1
            0x7FFF_FFFF, // >> 1
            0xFFFF_FFC0, // << 6
            0x03FF_FFFF, // >> 6
            0xFFFF_FFFF, // << 32
            0xFFFF_FFFF, // >> 32
            // S32
            0xFFFF_FFFF, // << 0
            0xFFFF_FFFF, // >> 0
            0xFFFF_FFFE, // << 1
            0xFFFF_FFFF, // >> 1
            0xFFFF_FFC0, // << 6
            0xFFFF_FFFF, // >> 6
            0xFFFF_FFFF, // << 32
            0xFFFF_FFFF, // >> 32
        ],
    );
}

#[test]
fn constant_shifts_64() {
    let results: Vec<u64> = vec![0; 20];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::U64]);
    let result_ptr = block.input(0);
    let input = block.input(1);
    let mut index = 0;
    for tp in vec![DataType::U64, DataType::S64] {
        for const_shift_amount in vec![0, 1, 6, 32, 64] {
            let left_result = block.left_shift(tp, input, const_u32(const_shift_amount));
            let right_result = block.right_shift(tp, input, const_u32(const_shift_amount));

            block.write_ptr(tp, result_ptr, index * size_of::<u64>(), left_result.val());
            index += 1;
            block.write_ptr(tp, result_ptr, index * size_of::<u64>(), right_result.val());
            index += 1;
        }
    }

    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, u64) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize, 2);
    println!("Shift 2: Results: {:?}", results);
    validate(
        &results,
        &[
            // U64
            2,           // << 0
            2,           // >> 0
            4,           // << 1
            1,           // >> 1
            128,         // << 6
            0,           // >> 6
            0x200000000, // << 32
            0,           // >> 32
            2,           // << 64
            2,           // >> 64
            // S64
            2,           // << 0
            2,           // >> 0
            4,           // << 1
            1,           // >> 1
            128,         // << 6
            0,           // >> 6
            0x200000000, // << 32
            0,           // >> 32
            2,           // << 64
            2,           // >> 64
        ],
    );

    f(results.as_ptr() as usize, 0xFFFF_FFFF_FFFF_FFFF);
    println!("Shift 0xFFFF_FFFF_FFFF_FFFF: Results: "); //{:?}", results);
    for r in results.iter() {
        print!("{:016X} ", r);
    }
    println!();
    validate(
        &results,
        &[
            // U64
            0xFFFFFFFFFFFFFFFF, // << 0
            0xFFFFFFFFFFFFFFFF, // >> 0
            0xFFFFFFFFFFFFFFFE, // << 1
            0x7FFFFFFFFFFFFFFF, // >> 1
            0xFFFFFFFFFFFFFFC0, // << 6
            0x03FFFFFFFFFFFFFF, // >> 6
            0xFFFFFFFF00000000, // << 32
            0x00000000FFFFFFFF, // >> 32
            0xFFFFFFFFFFFFFFFF, // << 64
            0xFFFFFFFFFFFFFFFF, // >> 64
            // S64
            0xFFFFFFFFFFFFFFFF, // << 0
            0xFFFFFFFFFFFFFFFF, // >> 0
            0xFFFFFFFFFFFFFFFE, // << 1
            0xFFFFFFFFFFFFFFFF, // >> 1
            0xFFFFFFFFFFFFFFC0, // << 6
            0xFFFFFFFFFFFFFFFF, // >> 6
            0xFFFFFFFF00000000, // << 32
            0xFFFFFFFFFFFFFFFF, // >> 32
            0xFFFFFFFFFFFFFFFF, // << 64
            0xFFFFFFFFFFFFFFFF, // >> 64
        ],
    );
}

#[test]
fn compiler_same_results_as_interpreter() {
    #[derive(Debug)]
    struct ResultStruct {
        pre_loop: u32,
        post_loop: u32,
    }
    fn get_function() -> IRFunction {
        let context = IRContext::new();
        let func = IRFunction::new(context);
        let mut first_block = func.new_block(vec![DataType::Ptr]);
        let result_ptr = first_block.input(0);
        let mut block = func.new_block(vec![]);

        let add_result = first_block.add(DataType::U32, const_u32(1), const_u32(1));
        let add2_result = first_block.add(DataType::U32, add_result.val(), const_u32(1));
        let add3_result = first_block.add(DataType::U32, add2_result.val(), add_result.val());
        let add4_result = first_block.add(DataType::U32, add3_result.val(), const_u32(1));

        first_block.jump(block.call(vec![]));

        // Very high register pressure
        let r1 = block.add(DataType::U32, add4_result.val(), const_u32(1));
        let r2 = block.add(DataType::U32, add4_result.val(), const_u32(1));
        let r3 = block.add(DataType::U32, add4_result.val(), const_u32(1));
        let r4 = block.add(DataType::U32, add4_result.val(), const_u32(1));
        let r5 = block.add(DataType::U32, add4_result.val(), const_u32(1));
        let r6 = block.add(DataType::U32, add4_result.val(), const_u32(1));
        let r7 = block.add(DataType::U32, r6.val(), r5.val());
        let r8 = block.add(DataType::U32, r7.val(), r4.val());
        let r9 = block.add(DataType::U32, r8.val(), r3.val());
        let r10 = block.add(DataType::U32, r9.val(), r2.val());
        let r11 = block.add(DataType::U32, r10.val(), r1.val());
        let nearly_final_result = block.add(DataType::U32, r11.val(), add4_result.val());
        block.write_ptr(DataType::U32, result_ptr, offset_of!(ResultStruct, pre_loop), nearly_final_result.val());

        // Use a loop to add ten to the final result
        let mut loop_block = func.new_block(vec![DataType::U32, DataType::U32]);
        block.jump(loop_block.call(vec![const_u32(0), nearly_final_result.val()]));

        // Add 1 to both the counter and the running total
        let loop_counter = loop_block.add(DataType::U32, loop_block.input(0), const_u32(1));
        let running_total = loop_block.add(DataType::U32, loop_block.input(1), const_u32(1));

        let loop_again = loop_block.compare(DataType::U32, loop_counter.val(), CompareType::LessThan, const_u32(10));
        let mut ret_block = func.new_block(vec![DataType::U32]);
        loop_block.branch(
            loop_again.val(),
            loop_block.call(vec![loop_counter.val(), running_total.val()]),
            ret_block.call(vec![running_total.val()]),
        );

        ret_block.write_ptr(DataType::U32, result_ptr, offset_of!(ResultStruct, post_loop), ret_block.input(0));
        ret_block.ret(Some(ret_block.input(0)));

        return func;
    }
    let func = get_function();

    let r = ResultStruct {
        pre_loop: 0,
        post_loop: 0,
    };

    println!("{}", func);
    println!("Interpreting");
    interpret_func(&func, vec![Constant::Ptr(&r as *const ResultStruct as usize)]);
    println!("Result: {:?}", r);

    println!("Compiling...");
    let compiled = compile(&func);

    let f: extern "C" fn(usize) -> u32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble_function(&compiled));

    println!("Running compiled code...");
    let r2 = ResultStruct {
        pre_loop: 0,
        post_loop: 0,
    };
    let retval = f(&r2 as *const ResultStruct as usize);

    println!("\n\nSummary:");
    println!("Interpreter result: {:?}", r);
    println!("   Compiler result: {:?}", r2);
    println!("Compiled function return value: {}", retval);
}

#[test]
fn convert_u32_u64() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::U32]);
    let converted = block.convert(DataType::U64, block.input(0));
    block.ret(Some(converted.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&func);
    let f: extern "C" fn(u32) -> u64 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble_function(&compiled));

    for i in 0..100 {
        assert_eq!(f(i), i as u64, "Failed to convert {} to u64", i);
    }
    assert_eq!(f(0x7FFFFFFF), 0x000000007FFFFFFF);
    assert_eq!(f(0xFFFFFFFF), 0x00000000FFFFFFFF);
    assert_eq!(f(0x80000000), 0x0000000080000000);
}

#[test]
fn convert_s32_s64() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::S32]);
    let converted = block.convert(DataType::S64, block.input(0));
    block.ret(Some(converted.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&func);
    let f: extern "C" fn(i32) -> i64 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble_function(&compiled));

    for i in 0..100 {
        assert_eq!(f(i), i as i64, "Failed to convert {} to i64", i);
    }
    assert_eq!(f(0x7FFFFFFF), 0x7FFFFFFF);
    assert_eq!(f(-1), -1);
    assert_eq!(f(-2147483648), -2147483648);
}

#[test]
fn call_external_function() {
    extern "C" fn add_ten(x: u32) -> u32 {
        x + 10
    }

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::U32]);

    let input = block.input(0);
    let add_ten_fn = external_fn!(add_ten(_));
    let call_result = block.call_function(add_ten_fn, &[input]);
    block.ret(Some(call_result.val()));

    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&func);
    let f: extern "C" fn(u32) -> u32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble_function(&compiled));

    println!("Running compiled code...");
    assert_eq!(f(0), 10);
    assert_eq!(f(1), 11);
    assert_eq!(f(2), 12);
}

const U128_VALUES: [u128; 6] = [
    0,
    u128::MAX,
    0x0123456789ABCDEF_FEDCBA9876543210,
    0xFFFFFFFFFFFFFFFF_0000000000000000,
    0x0000000000000000_FFFFFFFFFFFFFFFF,
    0x8000000000000000_0000000000000001,
];

/// There is no 128 bit constant yet, so build one from two 64 bit halves.
fn u128_const(block: &mut IRBlockHandle, value: u128) -> InputSlot {
    let high = block.vector_left_shift_bytes(DataType::U128, const_u64((value >> 64) as u64), const_u32(8));
    block.or(DataType::U128, high.val(), const_u64(value as u64)).val()
}

/// WritePtr has no 128 bit form yet, so store the halves into two consecutive u64 slots.
fn write_u128(block: &mut IRBlockHandle, result_ptr: InputSlot, index: usize, value: InputSlot) {
    let high = block.vector_right_shift_bytes(DataType::U128, value, const_u32(8));
    block.write_ptr(DataType::U64, result_ptr, index * size_of::<u64>(), value);
    block.write_ptr(DataType::U64, result_ptr, (index + 1) * size_of::<u64>(), high.val());
}

fn split_u128(values: &[u128]) -> Vec<u64> {
    values.iter().flat_map(|v| [*v as u64, (*v >> 64) as u64]).collect()
}

#[test]
fn simd_u128_roundtrip() {
    let results: Vec<u64> = vec![0; U128_VALUES.len() * 2];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr]);
    let result_ptr = block.input(0);

    for (i, value) in U128_VALUES.iter().enumerate() {
        let v = u128_const(&mut block, *value);
        write_u128(&mut block, result_ptr, i * 2, v);
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize);
    validate(&results, &split_u128(&U128_VALUES));
}

#[test]
fn simd_u128_not() {
    let results: Vec<u64> = vec![0; U128_VALUES.len() * 2];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr]);
    let result_ptr = block.input(0);

    for (i, value) in U128_VALUES.iter().enumerate() {
        let v = u128_const(&mut block, *value);
        let notted = block.not(DataType::U128, v);
        write_u128(&mut block, result_ptr, i * 2, notted.val());
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize);
    let expected = U128_VALUES.iter().map(|v| !*v).collect::<Vec<_>>();
    validate(&results, &split_u128(&expected));
}

#[test]
fn simd_u128_and() {
    let pairs = U128_VALUES
        .iter()
        .flat_map(|a| U128_VALUES.iter().map(move |b| (*a, *b)))
        .collect::<Vec<_>>();
    let results: Vec<u64> = vec![0; pairs.len() * 2];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr]);
    let result_ptr = block.input(0);

    for (i, (a, b)) in pairs.iter().enumerate() {
        let a = u128_const(&mut block, *a);
        let b = u128_const(&mut block, *b);
        let anded = block.and(DataType::U128, a, b);
        write_u128(&mut block, result_ptr, i * 2, anded.val());
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize);
    let expected = pairs.iter().map(|(a, b)| a & b).collect::<Vec<_>>();
    validate(&results, &split_u128(&expected));
}

#[test]
fn simd_u128_or() {
    let pairs = U128_VALUES
        .iter()
        .flat_map(|a| U128_VALUES.iter().map(move |b| (*a, *b)))
        .collect::<Vec<_>>();
    let results: Vec<u64> = vec![0; pairs.len() * 2];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr]);
    let result_ptr = block.input(0);

    for (i, (a, b)) in pairs.iter().enumerate() {
        let a = u128_const(&mut block, *a);
        let b = u128_const(&mut block, *b);
        let ored = block.or(DataType::U128, a, b);
        write_u128(&mut block, result_ptr, i * 2, ored.val());
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize);
    let expected = pairs.iter().map(|(a, b)| a | b).collect::<Vec<_>>();
    validate(&results, &split_u128(&expected));
}

/// Replaces an 8 byte window of a 128 bit value, exercising and/or/not together in a common way
#[test]
fn simd_u128_mask_merge() {
    let reg: u128 = 0x0011223344556677_8899AABBCCDDEEFF;
    let loaded: u128 = 0xA0A1A2A3A4A5A6A7;

    let results: Vec<u64> = vec![0; 16 * 2];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr]);
    let result_ptr = block.input(0);

    for e in 0..16u32 {
        let mask = block.vector_left_shift_bytes(DataType::U128, const_u64(u64::MAX), const_u32(e));
        let inv_mask = block.not(DataType::U128, mask.val());
        let r = u128_const(&mut block, reg);
        let masked = block.and(DataType::U128, r, inv_mask.val());
        let placed = block.vector_left_shift_bytes(DataType::U128, const_u64(loaded as u64), const_u32(e));
        let merged = block.or(DataType::U128, masked.val(), placed.val());
        write_u128(&mut block, result_ptr, e as usize * 2, merged.val());
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize);

    let expected = (0..16u32)
        .map(|e| {
            let mask = (u64::MAX as u128) << (e * 8);
            (reg & !mask) | (loaded << (e * 8))
        })
        .collect::<Vec<_>>();
    validate(&results, &split_u128(&expected));
}

#[test]
fn simd_u128_load_write_ptr() {
    let src: Vec<u128> = U128_VALUES.to_vec();
    let dst: Vec<u128> = vec![0; U128_VALUES.len()];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src_ptr = block.input(0);
    let dst_ptr = block.input(1);

    for i in 0..U128_VALUES.len() {
        let value = block.load_ptr(DataType::U128, src_ptr, i * size_of::<u128>());
        block.write_ptr(DataType::U128, dst_ptr, i * size_of::<u128>(), value.val());
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(src.as_ptr() as usize, dst.as_ptr() as usize);
    validate(&dst, &src);
}

/// 128 bit loads and stores must not require a 16 byte aligned address.
#[test]
fn simd_u128_load_write_ptr_unaligned() {
    const OFFSET: usize = 1;
    let mut src = vec![0u8; U128_VALUES.len() * size_of::<u128>() + OFFSET];
    for (i, value) in U128_VALUES.iter().enumerate() {
        let start = OFFSET + i * size_of::<u128>();
        src[start..start + size_of::<u128>()].copy_from_slice(&value.to_le_bytes());
    }
    let dst = vec![0u8; U128_VALUES.len() * size_of::<u128>() + OFFSET];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src_ptr = block.input(0);
    let dst_ptr = block.input(1);

    for i in 0..U128_VALUES.len() {
        let offset = OFFSET + i * size_of::<u128>();
        let value = block.load_ptr(DataType::U128, src_ptr, offset);
        block.write_ptr(DataType::U128, dst_ptr, offset, value.val());
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(src.as_ptr() as usize, dst.as_ptr() as usize);
    validate(&dst[OFFSET..], &src[OFFSET..]);
    assert_eq!(dst[0], 0, "Wrote before the start of the destination buffer");
}

/// Keeps more 128 bit values live at once than there are SIMD registers, forcing the
/// register allocator to spill them to the stack and reload them.
#[test]
fn simd_u128_spill() {
    // Derived from the register set rather than hardcoded, so this keeps forcing spills on
    // targets with a different number of SIMD registers.
    let simd_regs = get_registers().iter().filter(|r| r.is_simd()).count();
    assert!(simd_regs > 0, "Target has no SIMD registers to allocate");
    let count = simd_regs * 3;

    let values: Vec<u128> = (0..count)
        .map(|i| ((0xF00D_0000 + i as u128) << 64) | (0xBEEF_0000 + i as u128))
        .collect();
    let results: Vec<u128> = vec![0; count];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr]);
    let result_ptr = block.input(0);

    let live = values.iter().map(|v| u128_const(&mut block, *v)).collect::<Vec<_>>();
    for (i, value) in live.iter().enumerate() {
        block.write_ptr(DataType::U128, result_ptr, i * size_of::<u128>(), *value);
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(results.as_ptr() as usize);
    validate(&results, &values);
}

/// A value held in a volatile register must survive a call. The call is placed in a
/// non-entry block so that its position within the block differs from its index in the
/// function, and the callee clobbers every allocatable SIMD register.
#[test]
fn volatile_reg_live_across_call_in_later_block() {
    extern "C" fn clobber(x: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let junk = f32::from_bits(0x7F7F_7F7F);
            core::arch::asm!(
                "",
                inout("xmm8") junk => _,
                inout("xmm9") junk => _,
                inout("xmm10") junk => _,
                inout("xmm11") junk => _,
                inout("xmm12") junk => _,
                inout("xmm13") junk => _,
                inout("xmm14") junk => _,
                inout("xmm15") junk => _,
                options(nostack, preserves_flags)
            );
        }
        x.wrapping_add(1)
    }

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut entry = func.new_block(vec![DataType::Ptr, DataType::U64, DataType::F32]);
    let ptr = entry.input(0);
    let x = entry.input(1);
    let f = entry.input(2);

    // Pad so that function-level instruction indices diverge from in-block positions.
    let mut acc = x;
    for i in 0..6 {
        acc = entry.add(DataType::U64, acc, const_u64(i)).val();
    }
    let g = entry.add(DataType::F32, f, const_f32(1.0)).val();

    let mut block = func.new_block(vec![DataType::Ptr, DataType::U64, DataType::F32]);
    entry.jump(block.call(vec![ptr, acc, g]));

    let b_ptr = block.input(0);
    let b_acc = block.input(1);
    let b_f = block.input(2);
    let called = block.call_function(external_fn!(clobber(_)), &[b_acc]).val();
    // b_f is live across the call above and dies right here.
    let f_sum = block.add(DataType::F32, b_f, const_f32(2.0)).val();
    let mut pad = called;
    for i in 0..6 {
        pad = block.add(DataType::U64, pad, const_u64(i)).val();
    }
    block.write_ptr(DataType::U64, b_ptr, 0, pad);
    block.write_ptr(DataType::F32, b_ptr, 8, f_sum);
    block.ret(None);

    let compiled = compile(&func);
    let compiled_fn: extern "C" fn(usize, u64, f32) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    #[repr(C)]
    struct Results {
        int_result: u64,
        float_result: f32,
    }
    let results = Results {
        int_result: 0,
        float_result: 0.0,
    };
    compiled_fn(&results as *const Results as usize, 10, 5.0);

    let expected_int = {
        let mut a = 10u64;
        for i in 0..6 {
            a = a.wrapping_add(i);
        }
        let mut p = clobber(a);
        for i in 0..6 {
            p = p.wrapping_add(i);
        }
        p
    };
    assert_eq!(results.int_result, expected_int);
    assert_eq!(results.float_result, 5.0 + 1.0 + 2.0);
}

/// Unused parameters must still consume their ABI argument register, otherwise every
/// later argument is read from the wrong register.
#[test]
fn unused_function_arguments_dont_shift_later_ones() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::U64, DataType::U64, DataType::U64]);
    let third = block.input(2);
    let result = block.add(DataType::U64, third, const_u64(1)).val();
    block.ret(Some(result));

    let compiled = compile(&func);
    let f: extern "C" fn(u64, u64, u64) -> u64 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    assert_eq!(f(111, 222, 7), 8);
}

/// Entry block arguments must land in the registers the calling convention assigns to their
/// positions, which for interleaved classes differs between Windows x64 and SYSTEM-V/AAPCS.
#[test]
fn function_arguments_mixed_int_and_float() {
    #[derive(Default)]
    #[repr(C)]
    struct MixedArgs {
        int: u64,
        first_float: f32,
        second_float: f32,
    }

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::F32, DataType::U64, DataType::F32]);
    let ptr = block.input(0);
    let first_float = block.input(1);
    let int = block.input(2);
    let second_float = block.input(3);

    block.write_ptr(DataType::U64, ptr, offset_of!(MixedArgs, int), int);
    block.write_ptr(DataType::F32, ptr, offset_of!(MixedArgs, first_float), first_float);
    block.write_ptr(DataType::F32, ptr, offset_of!(MixedArgs, second_float), second_float);
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, f32, u64, f32) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    let results = MixedArgs::default();
    f(&results as *const MixedArgs as usize, 1.5, 42, 2.25);
    assert_eq!(results.int, 42);
    assert_eq!(results.first_float, 1.5);
    assert_eq!(results.second_float, 2.25);
}

/// Outgoing calls follow the same rule: float arguments go in the registers the callee reads them
/// from, and a float return value comes back in a SIMD register rather than the integer one.
#[test]
fn call_external_function_with_float_arguments() {
    extern "C" fn combine(x: u64, a: f32, b: f32) -> f32 {
        x as f32 + a * 10.0 + b * 100.0
    }

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::U64, DataType::F32]);
    let x = block.input(0);
    let a = block.input(1);
    // A constant argument has to be materialized into its argument register.
    let result = block.call_function(external_fn!(combine(_, _, _)), &[x, a, const_f32(0.5)]);
    block.ret(Some(result.val()));

    let compiled = compile(&func);
    let f: extern "C" fn(u64, f32) -> f32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    assert_eq!(f(3, 2.0), combine(3, 2.0, 0.5));
    assert_eq!(f(0, 0.0), combine(0, 0.0, 0.5));
    assert_eq!(f(7, -1.5), combine(7, -1.5, 0.5));
}

/// A float argument that is never read still consumes its argument register.
#[test]
fn unused_float_arguments_dont_shift_later_ones() {
    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::F32, DataType::U64, DataType::F32]);
    let third = block.input(2);
    let result = block.add(DataType::F32, third, const_f32(1.0)).val();
    block.ret(Some(result));

    let compiled = compile(&func);
    let f: extern "C" fn(f32, u64, f32) -> f32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    assert_eq!(f(111.0, 222, 7.0), 8.0);
}

/// Vector byte shifts by an amount that is only known at runtime, which take a different
/// code path than the constant amount shifts.
#[test]
fn simd_u128_variable_shifts() {
    let results: Vec<u64> = vec![0; U128_VALUES.len() * 4];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::U32]);
    let result_ptr = block.input(0);
    let bytes = block.input(1);

    let mut index = 0;
    for value in U128_VALUES.iter() {
        // Deliberately reused for both shifts, so a shift that clobbers its input is caught.
        let v = u128_const(&mut block, *value);
        let left = block.vector_left_shift_bytes(DataType::U128, v, bytes);
        write_u128(&mut block, result_ptr, index, left.val());
        index += 2;
        let right = block.vector_right_shift_bytes(DataType::U128, v, bytes);
        write_u128(&mut block, result_ptr, index, right.val());
        index += 2;
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, u32) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    // 16 and 17 bytes shift everything out, and must not wrap back around to a small shift.
    for bytes in 0..=17u32 {
        f(results.as_ptr() as usize, bytes);

        let expected = U128_VALUES
            .iter()
            .flat_map(|v| {
                if bytes >= 16 {
                    [0, 0]
                } else {
                    [v << (bytes * 8), v >> (bytes * 8)]
                }
            })
            .collect::<Vec<_>>();
        println!("Shift of {} bytes", bytes);
        validate(&results, &split_u128(&expected));
    }
}

/// A 32 bit add zero extends its result into the host register, so comparing S32 values as 64 bit
/// quantities turns every negative number into a large positive one. Covers each signed compare
/// against both a register and a constant.
#[test]
fn signed_32bit_compare_with_negative_values() {
    const CASES: [(i32, i32); 7] = [
        (-0x98, 0),
        (0, -0x98),
        (-1, -2),
        (-2, -1),
        (i32::MIN, i32::MAX),
        (i32::MAX, i32::MIN),
        (5, 5),
    ];
    const CMPS: [(CompareType, fn(i32, i32) -> bool); 6] = [
        (CompareType::LessThan, |a, b| a < b),
        (CompareType::LessThanOrEqual, |a, b| a <= b),
        (CompareType::GreaterThan, |a, b| a > b),
        (CompareType::GreaterThanOrEqual, |a, b| a >= b),
        (CompareType::Equal, |a, b| a == b),
        (CompareType::NotEqual, |a, b| a != b),
    ];

    // Two values per case: one comparing two registers, one comparing a register to a constant.
    let results: Vec<u32> = vec![0xFF; CASES.len() * CMPS.len() * 2];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr]);
    let result_ptr = block.input(0);

    let mut index = 0;
    for (a, b) in CASES.iter() {
        // Produced by an add so the value goes through the same zero extending path a computed
        // value does, rather than arriving as a constant.
        let a_reg = block.add(DataType::S32, const_s32(*a), const_s32(0)).val();
        let b_reg = block.add(DataType::S32, const_s32(*b), const_s32(0)).val();
        for (cmp, _) in CMPS.iter() {
            let reg_reg = block.compare(DataType::S32, a_reg, *cmp, b_reg);
            block.write_ptr(DataType::U32, result_ptr, index * size_of::<u32>(), reg_reg.val());
            index += 1;

            let reg_const = block.compare(DataType::S32, a_reg, *cmp, const_s32(*b));
            block.write_ptr(DataType::U32, result_ptr, index * size_of::<u32>(), reg_const.val());
            index += 1;
        }
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));
    f(results.as_ptr() as usize);

    let mut expected = Vec::new();
    for (a, b) in CASES.iter() {
        for (_, apply) in CMPS.iter() {
            let want = apply(*a, *b) as u32;
            expected.push(want);
            expected.push(want);
        }
    }
    validate(&results, &expected);
}

/// Values narrower than 32 bits can arrive in a register with unrelated bits set above them, so a
/// comparison has to widen them itself instead of trusting whatever produced them.
#[test]
fn narrow_compare_ignores_bits_above_the_type() {
    // (data type, raw register contents for a and b, expected signed interpretation)
    let cases: Vec<(DataType, u32, u32, i64, i64)> = vec![
        // 0x7F is +127 as an S8, but the raw value is negative when read as 32 bits.
        (DataType::S8, 0xFFFF_FF7F, 0x0000_0000, 127, 0),
        // 0x80 is -128 as an S8, but the raw value is positive when read as 32 bits.
        (DataType::S8, 0x0000_0080, 0x0000_0000, -128, 0),
        (DataType::U8, 0xFFFF_FF01, 0x0000_0002, 1, 2),
        (DataType::S16, 0xFFFF_7FFF, 0x0000_0000, 32767, 0),
        (DataType::S16, 0x0000_8000, 0x0000_0000, -32768, 0),
        (DataType::U16, 0xFFFF_0001, 0x0000_0002, 1, 2),
        // A 32 bit add zero extends, so a negative S32 has clear bits above it.
        (DataType::S32, 0xFFFF_FF68, 0x0000_0000, -0x98, 0),
        (DataType::U32, 0xFFFF_FF68, 0x0000_0000, 0xFFFF_FF68, 0),
    ];
    let cmps: Vec<(CompareType, fn(i64, i64) -> bool)> = vec![
        (CompareType::LessThan, |a, b| a < b),
        (CompareType::LessThanOrEqual, |a, b| a <= b),
        (CompareType::GreaterThan, |a, b| a > b),
        (CompareType::GreaterThanOrEqual, |a, b| a >= b),
        (CompareType::Equal, |a, b| a == b),
        (CompareType::NotEqual, |a, b| a != b),
    ];

    let results: Vec<u32> = vec![0xFF; cases.len() * cmps.len()];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr]);
    let result_ptr = block.input(0);

    let mut index = 0;
    for (tp, raw_a, raw_b, _, _) in cases.iter() {
        // Built with a U32 add so the register really does hold the raw bits, rather than the
        // compiler being free to materialize an already narrowed constant.
        let a = block.add(DataType::U32, const_u32(*raw_a), const_u32(0)).val();
        let b = block.add(DataType::U32, const_u32(*raw_b), const_u32(0)).val();
        for (cmp, _) in cmps.iter() {
            let result = block.compare(*tp, a, *cmp, b);
            block.write_ptr(DataType::U32, result_ptr, index * size_of::<u32>(), result.val());
            index += 1;
        }
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));
    f(results.as_ptr() as usize);

    let mut expected = Vec::new();
    for (_, _, _, a, b) in cases.iter() {
        for (_, apply) in cmps.iter() {
            expected.push(apply(*a, *b) as u32);
        }
    }
    validate(&results, &expected);
}

/// Vector types are 16 bytes like U128, so they must work everywhere U128 does: as a value in a
/// SIMD register, through memory, through a stack spill, and through the bitwise and byte shift
/// operations that do not care about lane structure.
#[test]
fn vector_type_roundtrip() {
    let types = [
        DataType::VU8,
        DataType::VS8,
        DataType::VU16,
        DataType::VS16,
        DataType::VU32,
        DataType::VS32,
    ];
    for tp in types {
        assert_eq!(tp.size(), 16, "{} should be 16 bytes", tp);
        assert!(tp.is_vector());
    }

    let src: Vec<u128> = U128_VALUES.to_vec();
    let dst: Vec<u128> = vec![0; U128_VALUES.len()];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src_ptr = block.input(0);
    let dst_ptr = block.input(1);

    for (i, tp) in types.iter().cycle().take(U128_VALUES.len()).enumerate() {
        let offset = i * size_of::<u128>();
        let value = block.load_ptr(*tp, src_ptr, offset);
        // Shift out and back so the value travels through the byte shift path as a vector, and
        // mask with an all ones value so it goes through the bitwise path too.
        let shifted = block.vector_left_shift_bytes(*tp, value.val(), const_u32(4));
        let restored = block.vector_right_shift_bytes(*tp, shifted.val(), const_u32(4));
        let ones = block.not(*tp, const_u64(0));
        let kept = block.and(*tp, restored.val(), ones.val());
        let low = block.vector_right_shift_bytes(*tp, value.val(), const_u32(12));
        let low = block.vector_left_shift_bytes(*tp, low.val(), const_u32(12));
        let result = block.or(*tp, kept.val(), low.val());
        block.write_ptr(*tp, dst_ptr, offset, result.val());
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(src.as_ptr() as usize, dst.as_ptr() as usize);
    validate(&dst, &src);
}

/// Spilling has to work for vector types too, not just U128.
#[test]
fn vector_type_spill() {
    let simd_regs = get_registers().iter().filter(|r| r.is_simd()).count();
    let count = simd_regs * 3;
    let values: Vec<u128> = (0..count)
        .map(|i| ((0xDEC0_0000u128 + i as u128) << 64) | (0x1234_0000 + i as u128))
        .collect();
    let results: Vec<u128> = vec![0; count];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src_ptr = block.input(0);
    let dst_ptr = block.input(1);

    let live = (0..count)
        .map(|i| block.load_ptr(DataType::VS16, src_ptr, i * size_of::<u128>()).val())
        .collect::<Vec<_>>();
    for (i, value) in live.iter().enumerate() {
        block.write_ptr(DataType::VS16, dst_ptr, i * size_of::<u128>(), *value);
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(values.as_ptr() as usize, results.as_ptr() as usize);
    validate(&results, &values);
}

fn swizzle_pattern(lanes: &[u8]) -> u64 {
    lanes
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, l)| acc | ((*l as u64) << (4 * i)))
}

/// Lane swizzles across a range of broadcast and pairwise patterns.
#[test]
fn vector_swizzle_lanes() {
    // Written highest lane first, which is the reverse of the pattern nibble order.
    let selections: [[u8; 8]; 16] = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 2, 2, 4, 4, 6, 6],
        [1, 1, 3, 3, 5, 5, 7, 7],
        [0, 0, 0, 0, 4, 4, 4, 4],
        [1, 1, 1, 1, 5, 5, 5, 5],
        [2, 2, 2, 2, 6, 6, 6, 6],
        [3, 3, 3, 3, 7, 7, 7, 7],
        [0; 8],
        [1; 8],
        [2; 8],
        [3; 8],
        [4; 8],
        [5; 8],
        [6; 8],
        [7; 8],
    ];

    let patterns: Vec<u64> = selections
        .iter()
        .map(|els| {
            let lanes: Vec<u8> = (0..8).map(|lane| 7 - els[7 - lane]).collect();
            swizzle_pattern(&lanes)
        })
        .collect();

    // Mirrored to match, so lane 7 - i holds i.
    let input: u128 = 0x0000_0001_0002_0003_0004_0005_0006_0007;
    let results: Vec<u128> = vec![0; patterns.len()];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src_ptr = block.input(0);
    let dst_ptr = block.input(1);

    let value = block.load_ptr(DataType::VU16, src_ptr, 0).val();
    for (i, pattern) in patterns.iter().enumerate() {
        let swizzled = block.vector_swizzle(DataType::VU16, value, *pattern);
        block.write_ptr(DataType::VU16, dst_ptr, i * size_of::<u128>(), swizzled.val());
    }
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(&input as *const u128 as usize, results.as_ptr() as usize);

    // Each slot holds its own index, so the expected output is the selection table itself.
    let expected: Vec<u128> = selections
        .iter()
        .map(|els| (0..8).fold(0u128, |acc, i| acc | ((els[i] as u128) << (16 * (7 - i)))))
        .collect();
    for (i, (got, want)) in results.iter().zip(expected.iter()).enumerate() {
        assert_eq!(got, want, "pattern {}: got {:#034x}, want {:#034x}", i, got, want);
    }
}

/// Swizzles at other lane widths, and patterns that are not just broadcasts.
#[test]
fn vector_swizzle_lane_widths() {
    let input: u128 = 0x0F0E0D0C_0B0A0908_07060504_03020100;
    let results: Vec<u128> = vec![0; 3];

    let reverse8 = swizzle_pattern(&[15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    let reverse32 = swizzle_pattern(&[3, 2, 1, 0]);
    let broadcast32 = swizzle_pattern(&[2, 2, 2, 2]);

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src_ptr = block.input(0);
    let dst_ptr = block.input(1);

    let v8 = block.load_ptr(DataType::VU8, src_ptr, 0).val();
    let r = block.vector_swizzle(DataType::VU8, v8, reverse8);
    block.write_ptr(DataType::VU8, dst_ptr, 0, r.val());
    let v32 = block.load_ptr(DataType::VU32, src_ptr, 0).val();
    let r = block.vector_swizzle(DataType::VU32, v32, reverse32);
    block.write_ptr(DataType::VU32, dst_ptr, size_of::<u128>(), r.val());
    let r = block.vector_swizzle(DataType::VU32, v32, broadcast32);
    block.write_ptr(DataType::VU32, dst_ptr, 2 * size_of::<u128>(), r.val());
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    f(&input as *const u128 as usize, results.as_ptr() as usize);

    assert_eq!(results[0], 0x00010203_04050607_08090A0B_0C0D0E0F, "reverse bytes");
    assert_eq!(results[1], 0x03020100_07060504_0B0A0908_0F0E0D0C, "reverse words");
    assert_eq!(results[2], 0x0B0A0908_0B0A0908_0B0A0908_0B0A0908, "broadcast word 2");
}

/// Packed lane add, saturating add, subtract, and the equality masks. Together these detect a
/// carry out of each lane, which is how a wider add is built from narrow ones.
#[test]
fn vector_lane_arithmetic() {
    let a_lanes: [u16; 8] = [0x0000, 0x0001, 0x7FFF, 0x8000, 0xFFFF, 0x1234, 0xFFFE, 0xABCD];
    let b_lanes: [u16; 8] = [0x0000, 0xFFFF, 0x0001, 0x8000, 0x0001, 0x1000, 0x0003, 0x0007];

    let pack = |v: &[u16; 8]| {
        v.iter()
            .enumerate()
            .fold(0u128, |acc, (i, l)| acc | ((*l as u128) << (16 * i)))
    };
    let a = pack(&a_lanes);
    let b = pack(&b_lanes);
    let results: Vec<u128> = vec![0; 6];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src = block.input(0);
    let dst = block.input(1);

    let va = block.load_ptr(DataType::VU16, src, 0).val();
    let vb = block.load_ptr(DataType::VU16, src, size_of::<u128>()).val();

    let sum = block.add(DataType::VU16, va, vb);
    block.write_ptr(DataType::VU16, dst, 0, sum.val());
    let usat = block.saturating_add(DataType::VU16, va, vb);
    block.write_ptr(DataType::VU16, dst, size_of::<u128>(), usat.val());
    let ssat = block.saturating_add(DataType::VS16, va, vb);
    block.write_ptr(DataType::VS16, dst, 2 * size_of::<u128>(), ssat.val());
    let diff = block.subtract(DataType::VU16, va, vb);
    block.write_ptr(DataType::VU16, dst, 3 * size_of::<u128>(), diff.val());

    // Unsigned saturation only kicks in when the wrapping sum carried out, so these two masks
    // are the carry detector.
    let eq = block.compare(DataType::VU16, sum.val(), CompareType::Equal, usat.val());
    block.write_ptr(DataType::VU16, dst, 4 * size_of::<u128>(), eq.val());
    let ne = block.compare(DataType::VU16, sum.val(), CompareType::NotEqual, usat.val());
    block.write_ptr(DataType::VU16, dst, 5 * size_of::<u128>(), ne.val());
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    let src_buf = [a, b];
    f(src_buf.as_ptr() as usize, results.as_ptr() as usize);

    let expect = |g: &dyn Fn(u16, u16) -> u16| {
        (0..8).fold(0u128, |acc, i| acc | ((g(a_lanes[i], b_lanes[i]) as u128) << (16 * i)))
    };
    let carried = |x: u16, y: u16| (x as u32) + (y as u32) > 0xFFFF;
    validate(
        &results,
        &[
            expect(&|x, y| x.wrapping_add(y)),
            expect(&|x, y| x.saturating_add(y)),
            expect(&|x, y| (x as i16).saturating_add(y as i16) as u16),
            expect(&|x, y| x.wrapping_sub(y)),
            expect(&|x, y| if carried(x, y) { 0 } else { 0xFFFF }),
            expect(&|x, y| if carried(x, y) { 0xFFFF } else { 0 }),
        ],
    );
}

/// Packed 16 bit lane multiplies. `Combined` keeps the low half of each product, `High` the
/// upper half, with the lane class deciding whether the high half is signed or unsigned.
#[test]
fn vector_multiply_lanes() {
    let lanes: [u16; 8] = [0x0000, 0x0001, 0x7FFF, 0x8000, 0xFFFF, 0x1234, 0x00FF, 0xABCD];
    let b_lanes: [u16; 8] = [0x0003, 0xFFFF, 0x0002, 0x8000, 0xFFFF, 0x1000, 0x0100, 0x0007];

    let pack = |v: &[u16; 8]| {
        v.iter()
            .enumerate()
            .fold(0u128, |acc, (i, l)| acc | ((*l as u128) << (16 * i)))
    };
    let a = pack(&lanes);
    let b = pack(&b_lanes);
    let results: Vec<u128> = vec![0; 3];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src = block.input(0);
    let dst = block.input(1);

    let va = block.load_ptr(DataType::VS16, src, 0).val();
    let vb = block.load_ptr(DataType::VS16, src, size_of::<u128>()).val();

    let lo = block.multiply(DataType::VS16, DataType::VS16, MultiplyType::Combined, va, vb);
    block.write_ptr(DataType::VS16, dst, 0, lo.val());
    let hi_s = block.multiply(DataType::VS16, DataType::VS16, MultiplyType::High, va, vb);
    block.write_ptr(DataType::VS16, dst, size_of::<u128>(), hi_s.val());
    let hi_u = block.multiply(DataType::VU16, DataType::VU16, MultiplyType::High, va, vb);
    block.write_ptr(DataType::VU16, dst, 2 * size_of::<u128>(), hi_u.val());
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    let src_buf = [a, b];
    f(src_buf.as_ptr() as usize, results.as_ptr() as usize);

    let expect = |g: &dyn Fn(u16, u16) -> u16| {
        (0..8).fold(0u128, |acc, i| acc | ((g(lanes[i], b_lanes[i]) as u128) << (16 * i)))
    };
    validate(
        &results,
        &[
            expect(&|x, y| x.wrapping_mul(y)),
            expect(&|x, y| (((x as i16 as i32) * (y as i16 as i32)) >> 16) as u16),
            expect(&|x, y| (((x as u32) * (y as u32)) >> 16) as u16),
        ],
    );
}

/// The interleave lane widths and the narrower pack that the multiplies don't reach.
#[test]
fn vector_interleave_lane_widths() {
    const VU64: DataType = DataType::Vector(VectorType::new(LaneClass::Unsigned, 64, 2));
    let a: u128 = 0x0F0E0D0C_0B0A0908_07060504_03020100;
    let b: u128 = 0x1F1E1D1C_1B1A1918_17161514_13121110;

    let words: [i16; 8] = [0, 1, -1, 127, -128, 200, -200, 32767];
    let w = (0..8).fold(0u128, |acc, i| acc | ((words[i] as u16 as u128) << (16 * i)));

    let results: Vec<u128> = vec![0; 6];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src = block.input(0);
    let dst = block.input(1);

    let v8a = block.load_ptr(DataType::VU8, src, 0).val();
    let v8b = block.load_ptr(DataType::VU8, src, size_of::<u128>()).val();
    for (i, half) in [VectorHalf::Low, VectorHalf::High].iter().enumerate() {
        let r = block.vector_interleave(DataType::VU8, *half, v8a, v8b);
        block.write_ptr(DataType::VU8, dst, i * size_of::<u128>(), r.val());
    }

    let v32a = block.load_ptr(DataType::VU32, src, 0).val();
    let v32b = block.load_ptr(DataType::VU32, src, size_of::<u128>()).val();
    let r = block.vector_interleave(DataType::VU32, VectorHalf::Low, v32a, v32b);
    block.write_ptr(DataType::VU32, dst, 2 * size_of::<u128>(), r.val());

    let v64a = block.load_ptr(VU64, src, 0).val();
    let v64b = block.load_ptr(VU64, src, size_of::<u128>()).val();
    let r = block.vector_interleave(VU64, VectorHalf::Low, v64a, v64b);
    block.write_ptr(VU64, dst, 3 * size_of::<u128>(), r.val());

    let v16 = block.load_ptr(DataType::VS16, src, 2 * size_of::<u128>()).val();
    let s = block.vector_pack(DataType::VS8, PackType::Saturating, v16, v16);
    block.write_ptr(DataType::VS8, dst, 4 * size_of::<u128>(), s.val());
    let u = block.vector_pack(DataType::VU8, PackType::Saturating, v16, v16);
    block.write_ptr(DataType::VU8, dst, 5 * size_of::<u128>(), u.val());
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    let src_buf = [a, b, w];
    f(src_buf.as_ptr() as usize, results.as_ptr() as usize);

    let lane = |v: u128, bits: u32, i: u32| (v >> (bits * i)) & ((1u128 << bits) - 1);
    let build = |bits: u32, f: &dyn Fn(u32) -> u128| (0..(128 / bits)).fold(0u128, |acc, i| acc | (f(i) << (bits * i)));
    // Each half reads only half of each input, so lane i of the result alternates a and b.
    let interleave = |bits: u32, high: bool| {
        let base = if high { (128 / bits) / 2 } else { 0 };
        build(bits, &|i| {
            let src = if i % 2 == 0 { a } else { b };
            lane(src, bits, base + i / 2)
        })
    };
    let packed = |f: &dyn Fn(i16) -> u8| build(8, &|i| f(words[(i % 8) as usize]) as u128);

    validate(
        &results,
        &[
            interleave(8, false),
            interleave(8, true),
            interleave(32, false),
            interleave(64, false),
            packed(&|v| v.clamp(-128, 127) as i8 as u8),
            packed(&|v| v.clamp(0, 255) as u8),
        ],
    );
}

/// Interleaves 16 bit halves back into 32 bit values and saturates them to 16 bits.
#[test]
fn vector_interleave_and_pack() {
    let values: [i32; 8] = [0, 1, -1, 32767, -32768, 40000, -40000, 0x7FFFFFFF];
    let pack = |g: &dyn Fn(i32) -> u16| (0..8).fold(0u128, |acc, i| acc | ((g(values[i]) as u128) << (16 * i)));
    let low = pack(&|v| v as u16);
    let high = pack(&|v| (v >> 16) as u16);
    let results: Vec<u128> = vec![0; 2];

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![DataType::Ptr, DataType::Ptr]);
    let src = block.input(0);
    let dst = block.input(1);

    let vh = block.load_ptr(DataType::VS16, src, 0).val();
    let vl = block.load_ptr(DataType::VS16, src, size_of::<u128>()).val();
    let lo32 = block.vector_interleave(DataType::VS32, VectorHalf::Low, vl, vh).val();
    let hi32 = block.vector_interleave(DataType::VS32, VectorHalf::High, vl, vh).val();
    let s = block.vector_pack(DataType::VS16, PackType::Saturating, lo32, hi32);
    block.write_ptr(DataType::VS16, dst, 0, s.val());
    let u = block.vector_pack(DataType::VU16, PackType::Saturating, lo32, hi32);
    block.write_ptr(DataType::VU16, dst, size_of::<u128>(), u.val());
    block.ret(None);

    let compiled = compile(&func);
    let f: extern "C" fn(usize, usize) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble_function(&compiled));

    let src_buf = [high, low];
    f(src_buf.as_ptr() as usize, results.as_ptr() as usize);

    validate(
        &results,
        &[
            pack(&|v| v.clamp(-32768, 32767) as u16),
            pack(&|v| v.clamp(0, 65535) as u16),
        ],
    );
}
