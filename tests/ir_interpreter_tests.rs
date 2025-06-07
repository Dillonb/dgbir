use dgbir::{ir::*, ir_interpreter::interpret_func};

#[test]
fn write_ptr() {
    let r: u32 = 0;

    let context = IRContext::new();
    let func = IRFunction::new(context);

    let mut block = func.new_block(vec![]);

    block.write_ptr(DataType::U32, const_ptr(&r as *const u32 as usize), 0, const_u32(1));

    block.ret(None);
    println!("{}", func);
    interpret_func(&func, vec![]);
    assert_eq!(r, 1);
}

#[test]
fn add_write_ptr() {
    let r: u32 = 0;

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![]);

    let add_result = block.add(DataType::U32, const_u32(1), const_u32(1));
    let add2_result = block.add(DataType::U32, add_result.val(), const_u32(1));
    let add3_result = block.add(DataType::U32, add2_result.val(), add_result.val());

    block.write_ptr(DataType::U32, const_ptr(&r as *const u32 as usize), 0, add3_result.val());
    block.ret(None);
    println!("{}", func);
    interpret_func(&func, vec![]);
    assert_eq!(r, 5);
}

#[test]
fn write_float_ptr() {
    let r: f32 = 0.0;

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![]);

    block.write_ptr(DataType::F32, const_ptr(&r as *const f32 as usize), 0, const_f32(1.0));
    block.ret(None);

    println!("{}", func);

    interpret_func(&func, vec![]);
    assert_eq!(r, 1.0);
}

#[test]
fn add_write_float_ptr() {
    let res_1: f32 = 0.0;
    let res_2: u32 = 0;

    let context = IRContext::new();
    let func = IRFunction::new(context);
    let mut block = func.new_block(vec![]);

    let add_result = block.add(DataType::F32, const_f32(1.0), const_u32(1));
    let add_result_2 = block.add(DataType::U32, const_u32(1), const_f32(1.0));

    block.write_ptr(DataType::F32, const_ptr(&res_1 as *const f32 as usize), 0, add_result.val());

    block.write_ptr(DataType::U32, const_ptr(&res_2 as *const u32 as usize), 0, add_result_2.val());
    block.ret(None);

    println!("{}", func);

    interpret_func(&func, vec![]);
    assert_eq!(res_1, 2.0);
    assert_eq!(res_2, 2);
}

#[test]
fn test_conditional_branch_loop() {
    let res: u32 = 0;

    /*
     * int i = 0;
     * while (i < 10) {
     *   i++;
     * }
     */

    let context = IRContext::new();
    let func = IRFunction::new(context);

    let mut entry_block = func.new_block(vec![]);
    let mut loop_block = func.new_block(vec![DataType::U32]);
    // TODO: rename "call" to something that makes it clear we're not appending a call instruction
    // to the block, we're calling the block
    entry_block.jump(loop_block.call(vec![const_u32(0)]));

    let running_sum = loop_block.add(DataType::U32, loop_block.input(0), const_u32(1));

    let compare = loop_block.compare(running_sum.val(), CompareType::LessThanUnsigned, const_u32(10));

    let mut exit_block = func.new_block(vec![DataType::U32]);
    loop_block.branch(
        compare.val(),
        loop_block.call(vec![running_sum.val()]),
        exit_block.call(vec![running_sum.val()]),
    );

    exit_block.write_ptr(DataType::U32, const_ptr(&res as *const u32 as usize), 0, exit_block.input(0));

    exit_block.ret(Some(exit_block.input(0)));

    println!("{}", func);
    interpret_func(&func, vec![]);
    assert_eq!(res, 10);
}
