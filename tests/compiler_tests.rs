use std::mem::{self, offset_of};

use dgbir::{
    compiler::compile,
    disassembler::disassemble,
    ir::{const_f32, const_u32, CompareType, Constant, DataType, IRContext, IRFunction},
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
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::U32]);
    func.add(&block, DataType::U32, block.input(0), const_u32(1));
    // No return statement, block is unclosed
    println!("{}", func);
    println!("Compiling...");
    compile(&mut func);
}

#[test]
fn compiler_identityfunc() {
    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::U32]);
    func.ret(&block, Some(block.input(0)));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&mut func);
    let f: extern "C" fn(u32) -> u32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble(&compiled.code, f as u64));

    assert_eq!(f(0), 0);
    assert_eq!(f(1), 1);
    assert_eq!(f(2), 2);
    assert_eq!(f(10000), 10000);
}

#[test]
fn compiler_addone() {
    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::U32]);
    let add_result = func.add(&block, DataType::U32, block.input(0), const_u32(1));
    func.ret(&block, Some(add_result.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&mut func);
    let f: extern "C" fn(u32) -> u32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble(&compiled.code, f as u64));

    assert_eq!(f(0), 1);
    assert_eq!(f(1), 2);
    assert_eq!(f(2), 3);
    assert_eq!(f(10000), 10001);
}

#[test]
fn compiler_identityfunc_f32() {
    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::F32]);
    func.ret(&block, Some(block.input(0)));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&mut func);
    let f: extern "C" fn(f32) -> f32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble(&compiled.code, f as u64));

    assert_eq!(f(0.0), 0.0);
    assert_eq!(f(1.0), 1.0);
    assert_eq!(f(2.0), 2.0);
    assert_eq!(f(10000.0), 10000.0);
}

#[test]
fn compiler_addone_f32() {
    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::F32]);
    let add_result = func.add(&block, DataType::F32, block.input(0), const_f32(1.0));
    func.ret(&block, Some(add_result.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&mut func);
    let f: extern "C" fn(f32) -> f32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble(&compiled.code, f as u64));
    assert_eq!(f(0.0), 1.0);
    assert_eq!(f(1.0), 2.0);
    assert_eq!(f(2.0), 3.0);
    assert_eq!(f(10000.0), 10001.0);
}

#[test]
fn compiler_add_f32_to_self() {
    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::F32]);
    let add_result = func.add(&block, DataType::F32, block.input(0), block.input(0));
    func.ret(&block, Some(add_result.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&mut func);
    let f: extern "C" fn(f32) -> f32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble(&compiled.code, f as u64));
    assert_eq!(f(0.0), 0.0);
    assert_eq!(f(1.0), 2.0);
    assert_eq!(f(2.0), 4.0);
    assert_eq!(f(10000.0), 20000.0);
}

#[test]
fn constant_shifts_8() {
    let results: Vec<u64> = vec![0; 16];

    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::Ptr, DataType::U8]);
    let result_ptr = block.input(0);
    let input = block.input(1);
    let mut index = 0;
    for tp in vec![DataType::U8, DataType::S8] {
        for const_shift_amount in vec![0, 1, 6, 8] {
            let left_result = func.left_shift(&block, tp, input, const_u32(const_shift_amount));
            let right_result = func.right_shift(&block, tp, input, const_u32(const_shift_amount));

            // Write the entire 64 bit register even though we only did a 32 bit shift to ensure
            // the behavior is consistent
            func.write_ptr(&block, DataType::U64, result_ptr, index * size_of::<u64>(), left_result.val());
            index += 1;
            func.write_ptr(&block, DataType::U64, result_ptr, index * size_of::<u64>(), right_result.val());
            index += 1;
        }
    }

    func.ret(&block, None);

    let compiled = compile(&mut func);
    let f: extern "C" fn(usize, u8) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    // println!("{}", disassemble(&compiled.code, f as u64));

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
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::Ptr, DataType::U16]);
    let result_ptr = block.input(0);
    let input = block.input(1);
    let mut index = 0;
    for tp in vec![DataType::U16, DataType::S16] {
        for const_shift_amount in vec![0, 1, 6, 32] {
            let left_result = func.left_shift(&block, tp, input, const_u32(const_shift_amount));
            let right_result = func.right_shift(&block, tp, input, const_u32(const_shift_amount));

            // Write the entire 64 bit register even though we only did a 32 bit shift to ensure
            // the behavior is consistent
            func.write_ptr(&block, DataType::U64, result_ptr, index * size_of::<u64>(), left_result.val());
            index += 1;
            func.write_ptr(&block, DataType::U64, result_ptr, index * size_of::<u64>(), right_result.val());
            index += 1;
        }
    }

    func.ret(&block, None);

    let compiled = compile(&mut func);
    let f: extern "C" fn(usize, u16) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble(&compiled.code, f as u64));

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
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::Ptr, DataType::U32]);
    let result_ptr = block.input(0);
    let input = block.input(1);
    let mut index = 0;
    for tp in vec![DataType::U32, DataType::S32] {
        for const_shift_amount in vec![0, 1, 6, 32] {
            let left_result = func.left_shift(&block, tp, input, const_u32(const_shift_amount));
            let right_result = func.right_shift(&block, tp, input, const_u32(const_shift_amount));

            // Write the entire 64 bit register even though we only did a 32 bit shift to ensure
            // the behavior is consistent
            func.write_ptr(&block, DataType::U64, result_ptr, index * size_of::<u64>(), left_result.val());
            index += 1;
            func.write_ptr(&block, DataType::U64, result_ptr, index * size_of::<u64>(), right_result.val());
            index += 1;
        }
    }

    func.ret(&block, None);

    let compiled = compile(&mut func);
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
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::Ptr, DataType::U64]);
    let result_ptr = block.input(0);
    let input = block.input(1);
    let mut index = 0;
    for tp in vec![DataType::U64, DataType::S64] {
        for const_shift_amount in vec![0, 1, 6, 32, 64] {
            let left_result = func.left_shift(&block, tp, input, const_u32(const_shift_amount));
            let right_result = func.right_shift(&block, tp, input, const_u32(const_shift_amount));

            func.write_ptr(&block, tp, result_ptr, index * size_of::<u64>(), left_result.val());
            index += 1;
            func.write_ptr(&block, tp, result_ptr, index * size_of::<u64>(), right_result.val());
            index += 1;
        }
    }

    func.ret(&block, None);

    let compiled = compile(&mut func);
    let f: extern "C" fn(usize, u64) = unsafe { mem::transmute(compiled.ptr_entrypoint()) };
    println!("{}", disassemble(&compiled.code, f as u64));

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
        let mut func = IRFunction::new(context);
        let first_block = func.new_block(vec![DataType::Ptr]);
        let result_ptr = first_block.input(0);
        let block = func.new_block(vec![]);

        let add_result = func.add(&first_block, DataType::U32, const_u32(1), const_u32(1));
        let add2_result = func.add(&first_block, DataType::U32, add_result.val(), const_u32(1));
        let add3_result = func.add(&first_block, DataType::U32, add2_result.val(), add_result.val());
        let add4_result = func.add(&first_block, DataType::U32, add3_result.val(), const_u32(1));

        func.jump(&first_block, block.call(vec![]));

        // Very high register pressure
        let r1 = func.add(&block, DataType::U32, add4_result.val(), const_u32(1));
        let r2 = func.add(&block, DataType::U32, add4_result.val(), const_u32(1));
        let r3 = func.add(&block, DataType::U32, add4_result.val(), const_u32(1));
        let r4 = func.add(&block, DataType::U32, add4_result.val(), const_u32(1));
        let r5 = func.add(&block, DataType::U32, add4_result.val(), const_u32(1));
        let r6 = func.add(&block, DataType::U32, add4_result.val(), const_u32(1));
        let r7 = func.add(&block, DataType::U32, r6.val(), r5.val());
        let r8 = func.add(&block, DataType::U32, r7.val(), r4.val());
        let r9 = func.add(&block, DataType::U32, r8.val(), r3.val());
        let r10 = func.add(&block, DataType::U32, r9.val(), r2.val());
        let r11 = func.add(&block, DataType::U32, r10.val(), r1.val());
        let nearly_final_result = func.add(&block, DataType::U32, r11.val(), add4_result.val());
        func.write_ptr(
            &block,
            DataType::U32,
            result_ptr,
            offset_of!(ResultStruct, pre_loop),
            nearly_final_result.val(),
        );

        // Use a loop to add ten to the final result
        let loop_block = func.new_block(vec![DataType::U32, DataType::U32]);
        func.jump(&block, loop_block.call(vec![const_u32(0), nearly_final_result.val()]));

        // Add 1 to both the counter and the running total
        let loop_counter = func.add(&loop_block, DataType::U32, loop_block.input(0), const_u32(1));
        let running_total = func.add(&loop_block, DataType::U32, loop_block.input(1), const_u32(1));

        let loop_again = func.compare(&loop_block, loop_counter.val(), CompareType::LessThanUnsigned, const_u32(10));
        let ret_block = func.new_block(vec![DataType::U32]);
        func.branch(
            &loop_block,
            loop_again.val(),
            loop_block.call(vec![loop_counter.val(), running_total.val()]),
            ret_block.call(vec![running_total.val()]),
        );

        func.write_ptr(&ret_block, DataType::U32, result_ptr, offset_of!(ResultStruct, post_loop), ret_block.input(0));
        func.ret(&ret_block, Some(ret_block.input(0)));

        return func;
    }
    let mut func = get_function();

    let r = ResultStruct {
        pre_loop: 0,
        post_loop: 0,
    };

    println!("{}", func);
    println!("Interpreting");
    interpret_func(&func, vec![Constant::Ptr(&r as *const ResultStruct as usize)]);
    println!("Result: {:?}", r);

    println!("Compiling...");
    let compiled = compile(&mut func);

    let f: extern "C" fn(usize) -> u32 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble(&compiled.code, f as u64));

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
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::U32]);
    let converted = func.convert(&block, DataType::U64, block.input(0));
    func.ret(&block, Some(converted.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&mut func);
    let f: extern "C" fn(u32) -> u64 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble(&compiled.code, f as u64));

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
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![DataType::S32]);
    let converted = func.convert(&block, DataType::S64, block.input(0));
    func.ret(&block, Some(converted.val()));
    println!("{}", func);
    println!("Compiling...");
    let compiled = compile(&mut func);
    let f: extern "C" fn(i32) -> i64 = unsafe { mem::transmute(compiled.ptr_entrypoint()) };

    println!("{}", disassemble(&compiled.code, f as u64));

    for i in 0..100 {
        assert_eq!(f(i), i as i64, "Failed to convert {} to i64", i);
    }
    assert_eq!(f(0x7FFFFFFF), 0x7FFFFFFF);
    assert_eq!(f(-1), -1);
    assert_eq!(f(-2147483648), -2147483648);
}

