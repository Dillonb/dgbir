use std::mem::{self, offset_of};

use dgbir::{
    compiler::compile,
    disassembler::disassemble,
    ir::{const_f32, const_u32, CompareType, Constant, DataType, IRContext, IRFunction},
    ir_interpreter::interpret_func,
};

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
