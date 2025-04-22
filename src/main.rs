use dgbir::{ir::*, ir_interpreter::interpret_func, ir_tostring::func_tostring};

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
    let block = func.new_block(vec![]);

    let add_result = func[&block].add(DataType::U32, const_u32(1), const_u32(1));
    let add2_result = func[&block].add(DataType::U32, add_result.val(), const_u32(1));
    let add3_result = func[&block].add(DataType::U32, add2_result.val(), add_result.val());

    func[&block].write_ptr(
        DataType::U32,
        const_ptr(&r as *const u32 as usize),
        add3_result.val(),
    );

    func[&block].ret(None);

    println!("Block:");
    println!("{}", func_tostring(&func));
    interpret_func(&func);
    println!("Result: {:?}", r);
}

// COMPLEX INSTRUCTION SET COMPUTING
// HELL YEAH
