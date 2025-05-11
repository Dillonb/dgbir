use dgbir::{compiler::compile, ir::*, ir_interpreter::interpret_func};

fn main() {
    // let samples = [
    //     "hello:",
    //     "do_something()",
    //     "do_something(test)",
    //     "do_something(test1,test2)",
    //     "do_something(test1, test2)",
    //     "do_something(test1 ,test2)",
    //     "do_something(test1 , test2)",
    //     "result = do_something(test)",
    //     "result, result2 = do_something(test)",
    //     "result , result2 = do_something(test)",
    //     "result , result2,result3 ,result4 = do_something(test)",
    //     "result:u8 = do_something()",
    //     ".output(result) = do_something()",
    //     ".output(result:u8), .output2(result2: u16) = do_something()",
    //     ".output(result:u8) = do_something(do_somethingelse(v100))",
    // ];
    // for data in samples {
    //     match parse_statement(data) {
    //         Ok((leftover, res)) => {
    //             if !leftover.is_empty() {
    //                 println!("leftover: {}", leftover);
    //             }

    //             println!("{:?}", res);
    //         }
    //         Result::Err(e) => panic!("{}", e),
    //     }
    // }

    let r: u32 = 0;

    let context = IRContext::new();
    let mut func = IRFunction::new(context);
    let first_block = func.new_block(vec![]);
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
    let final_result = func.add(&block, DataType::U32, r11.val(), add4_result.val());

    func.write_ptr(
        &block,
        DataType::U32,
        const_ptr(&r as *const u32 as usize),
        final_result.val(),
    );

    func.ret(&block, None);

    println!("{}", func);
    println!("Interpreting");
    interpret_func(&func);
    println!("Result: {:?}", r);

    compile(&mut func);
}

// COMPLEX INSTRUCTION SET COMPUTING
// HELL YEAH
