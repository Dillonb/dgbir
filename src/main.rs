use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::char,
    combinator::{map, value},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, preceded, separated_pair, terminated},
    IResult, Parser,
};
use std::str;

#[derive(Debug, Clone)]
pub enum Type {
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    U64,
    S64,
    F32,
    F64
}

#[derive(Debug)]
pub enum Lhs {
    Value(String),
    NamedValue(String, String),
    TypedValue(Type, String),
    TypedNamedValue(Type, String, String)
}

#[derive(Debug)]
pub enum Rhs {
    IrOperation((String, Vec<Rhs>)),
    Value(String),
}

#[derive(Debug)]
pub enum Statement {
    Label(String),
    RhsOnly(Rhs),
    Assignment(Vec<Lhs>, Rhs)
}

fn whitespace(i: &str) -> IResult<&str, &str> {
    let chars = " \t";
    take_while(move |c| chars.contains(c))(i)
}

fn identifier(i: &str) -> IResult<&str, &str> {
    let chars = "abcdefghijklmnopqrstuvwxyz1234567890_";
    take_while1(move |c| chars.contains(c))(i)
}

fn label(i: &str) -> IResult<&str, Statement> {
    map(terminated(identifier, char(':')), |x: &str| {
        Statement::Label(x.to_string())
    })
    .parse(i)
}

fn function_arguments(i: &str) -> IResult<&str, Vec<Rhs>> {
    separated_list0((whitespace, char(','), whitespace), rhs).parse(i)
}

fn function_call(i: &str) -> IResult<&str, Rhs> {
    map(
        (
            identifier,
            delimited(char('('), function_arguments, char(')')),
        ),
        |(fn_name, args)| Rhs::IrOperation((fn_name.to_string(), args)),
    )
    .parse(i)
}

fn rhs(i: &str) -> IResult<&str, Rhs> {
    alt((
        function_call,
        map(identifier, |x| Rhs::Value(x.to_string())),
    ))
    .parse(i)
}

fn ir_type(i: &str) -> IResult<&str, Type> {
    alt((
        value(Type::U8, tag("u8")),
        value(Type::S8, tag("s8")),
        value(Type::U16, tag("u16")),
        value(Type::S16, tag("s16")),
        value(Type::U32, tag("u32")),
        value(Type::S32, tag("s32")),
        value(Type::U64, tag("u64")),
        value(Type::S64, tag("s64")),
    )).parse(i)
}

fn lhs(i: &str) -> IResult<&str, Vec<Lhs>> {
    // TODO: reduce duplication?
    let lhs_identifier = map(identifier, |x| Lhs::Value(x.to_string()));

    let typed_lhs_identifier = map(separated_pair(identifier, (whitespace, char(':'), whitespace), ir_type), |(id, tp)| Lhs::TypedValue(tp, id.to_string()));

    let named_lhs_identifier = map(
        preceded(
            char('.'),
            (
                terminated(identifier, whitespace),
                delimited(
                    char('('),
                    identifier,
                    char(')')
                )
            )
        ), |(name, id)| Lhs::NamedValue(name.to_string(), id.to_string()));

    let named_typed_lhs_identifier = map(
        preceded(
            char('.'),
            (
                terminated(identifier, whitespace),
                delimited(
                    char('('),
                    separated_pair(identifier, (whitespace, char(':'), whitespace), ir_type),
                    char(')')
                )
            )
        ), |(name, (id, tp))| Lhs::TypedNamedValue(tp, name.to_string(), id.to_string()));

    separated_list1(
        (whitespace, char(','), whitespace),
        alt((
                named_typed_lhs_identifier,
                named_lhs_identifier,
                typed_lhs_identifier,
                lhs_identifier,
        ))
    ).parse(i)
}

fn statement(i: &str) -> IResult<&str, Statement> {
    alt((
        map((lhs, (whitespace, char('='), whitespace), rhs), |(l, _, r)| Statement::Assignment(l, r)),
        label,
        map(rhs, Statement::RhsOnly),
    )).parse(i)
}

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
        ".output(result:u8) = do_something(do_somethingelse(v100))"
    ];
    for data in samples {
        match statement(data) {
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
