use dgbir::parser::parse_statement;

fn main() {
    let samples = [
        "hello:",
        "do_something()",
        "do_something(test)",
        "do_something(test1,test2)",
        "do_something(test1, test2)",
        "do_something(test1 ,test2)",
        "do_something(test1 , test2)",
        "result = do_something(test)",
        "result, result2 = do_something(test)",
        "result , result2 = do_something(test)",
        "result , result2,result3 ,result4 = do_something(test)",
        "result:u8 = do_something()",
        ".output(result) = do_something()",
        ".output(result:u8), .output2(result2: u16) = do_something()",
        ".output(result:u8) = do_something(do_somethingelse(v100))",
    ];
    for data in samples {
        match parse_statement(data) {
            Ok((leftover, res)) => {
                if !leftover.is_empty() {
                    println!("leftover: {}", leftover);
                }

                println!("{:?}", res);
            }
            Result::Err(e) => panic!("{}", e),
        }
    }
}

// COMPLEX INSTRUCTION SET COMPUTING
// HELL YEAH
