use dgbir::{ir::*, ir_interpreter::interpret_block};

#[test]
fn write_ptr() {
    let r: u32 = 0;

    let context = IRContext::new();
    let mut block = IRBlock::new(context);

    block.write_ptr(
        DataType::U32,
        IRBlock::const_ptr(&r as *const u32 as usize),
        IRBlock::const_u32(1),
    );

    interpret_block(&block);
    assert_eq!(r, 1);
}

#[test]
fn add_write_ptr() {
    let r: u32 = 0;

    let context = IRContext::new();
    let mut block = IRBlock::new(context);

    let add_result = block.add(DataType::U32, IRBlock::const_u32(1), IRBlock::const_u32(1));
    let add2_result = block.add(DataType::U32, add_result.val(), IRBlock::const_u32(1));
    let add3_result = block.add(DataType::U32, add2_result.val(), add_result.val());

    block.write_ptr(
        DataType::U32,
        IRBlock::const_ptr(&r as *const u32 as usize),
        add3_result.val(),
    );
    interpret_block(&block);
    assert_eq!(r, 5);
}

#[test]
fn write_float_ptr() {
    let r: f32 = 0.0;

    let context = IRContext::new();
    let mut block = IRBlock::new(context);

    block.write_ptr(
        DataType::F32,
        IRBlock::const_ptr(&r as *const f32 as usize),
        IRBlock::const_f32(1.0),
    );

    interpret_block(&block);
    assert_eq!(r, 1.0);
}

#[test]
fn add_write_float_ptr() {
    let res_1: f32 = 0.0;
    let res_2: u32 = 0;

    let context = IRContext::new();
    let mut block = IRBlock::new(context);

    let add_result = block.add(DataType::F32, IRBlock::const_f32(1.0), IRBlock::const_u32(1));
    let add_result_2 = block.add(DataType::U32, IRBlock::const_u32(1), IRBlock::const_f32(1.0));

    block.write_ptr(
        DataType::F32,
        IRBlock::const_ptr(&res_1 as *const f32 as usize),
        add_result.val(),
    );

    block.write_ptr(
        DataType::U32,
        IRBlock::const_ptr(&res_2 as *const u32 as usize),
        add_result_2.val(),
    );

    interpret_block(&block);
    assert_eq!(res_1, 2.0);
    assert_eq!(res_2, 2);
}
