use dgbir::{ir::*, ir_interpreter::interpret_func, ir_tostring::func_tostring};

#[test]
fn write_ptr() {
    let r: u32 = 0;

    let context = IRContext::new();
    let mut func = IRFunction::new(context);

    let block = func.new_block(vec![]);

    func[&block].write_ptr(
        DataType::U32,
        const_ptr(&r as *const u32 as usize),
        const_u32(1),
    );

    func[&block].ret(const_u32(1));

    interpret_func(&func);
    assert_eq!(r, 1);
}

#[test]
fn add_write_ptr() {
    let r: u32 = 0;

    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![]);

    let add_result = func[&block].add(DataType::U32, const_u32(1), const_u32(1));
    let add2_result = func[&block].add(DataType::U32, add_result.val(), const_u32(1));
    let add3_result = func[&block].add(DataType::U32, add2_result.val(), add_result.val());

    func[&block].write_ptr(
        DataType::U32,
        const_ptr(&r as *const u32 as usize),
        add3_result.val(),
    );
    func[&block].ret(const_u32(1));
    interpret_func(&func);
    assert_eq!(r, 5);
}

#[test]
fn write_float_ptr() {
    let r: f32 = 0.0;

    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![]);

    func[&block].write_ptr(
        DataType::F32,
        const_ptr(&r as *const f32 as usize),
        const_f32(1.0),
    );
    func[&block].ret(const_u32(1));

    interpret_func(&func);
    assert_eq!(r, 1.0);
}

#[test]
fn add_write_float_ptr() {
    let res_1: f32 = 0.0;
    let res_2: u32 = 0;

    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let block = func.new_block(vec![]);

    let add_result = func[&block].add(DataType::F32, const_f32(1.0), const_u32(1));
    let add_result_2 = func[&block].add(DataType::U32, const_u32(1), const_f32(1.0));

    func[&block].write_ptr(
        DataType::F32,
        const_ptr(&res_1 as *const f32 as usize),
        add_result.val(),
    );

    func[&block].write_ptr(
        DataType::U32,
        const_ptr(&res_2 as *const u32 as usize),
        add_result_2.val(),
    );
    func[&block].ret(const_u32(1));

    interpret_func(&func);
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
    let mut func = IRFunction::new(context);

    let entry_block = func.new_block(vec![]);
    let loop_block = func.new_block(vec![DataType::U32]);
    // TODO: rename "call" to something that makes it clear we're not appending a call instruction
    // to the block, we're calling the block
    func[&entry_block].jump(loop_block.call(vec![const_u32(0)]));

    let running_sum = func[&loop_block].add(DataType::U32, loop_block.input(0), const_u32(1));
    let compare = func[&loop_block].compare(
        running_sum.val(),
        CompareType::LessThanUnsigned,
        const_u32(10),
    );

    let exit_block = func.new_block(vec![DataType::U32]);
    func[&loop_block].branch(
        compare.val(),
        loop_block.call(vec![running_sum.val()]),
        exit_block.call(vec![running_sum.val()]),
    );

    func[&exit_block].write_ptr(
        DataType::U32,
        const_ptr(&res as *const u32 as usize),
        exit_block.input(0),
    );

    func[&exit_block].ret(exit_block.input(0));

    println!("{}", func_tostring(&func));
    interpret_func(&func);
    assert_eq!(res, 10);
}
