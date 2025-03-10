use dgbir::{compiler::compile, ir::*, ir_interpreter::interpret_block};

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
    let r_ptr: usize = &r as *const u32 as usize;

    let context = IRContext::new();
    let mut block = IRBlock::new(context);
    let add_result = block.add(DataType::U32, IRBlock::const_u32(1), IRBlock::const_u32(1));
    let add2_result = block.add(DataType::U32, add_result.at(0), IRBlock::const_u32(1));

    block.write_ptr(IRBlock::const_ptr(r_ptr), add2_result.at(0));

    println!("r: {:?}", r);

    // compile(block);
    interpret_block(block);
}

// COMPLEX INSTRUCTION SET COMPUTING
// HELL YEAH
