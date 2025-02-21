use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, multispace0, one_of},
    combinator::{recognize, value},
    error::ParseError,
    multi::{many0, many0_count, many1},
    sequence::{pair, terminated},
    IResult, Parser,
};

use crate::types::*;

pub fn parse(src: &str) -> ToyProgram {
    let (rest, program) = nom::Parser::parse(&mut parse_program, src).expect("Parse failure");
    if rest.trim() != "" {
        panic!("Parse failure. Unparsed suffix: {}", rest);
    }
    return program;
}

fn parse_program(input: &str) -> IResult<&str, ToyProgram> {
    (many0(parse_def), parse_expression)
        .map(|(defs, main)| ToyProgram { defs, main })
        .parse(input)
}

fn parse_def(input: &str) -> IResult<&str, ToyDef> {
    (parse_var, ws(tag("=")), parse_expression, ws(tag(";")))
        .map(|(name, _, body, _)| ToyDef { name, body })
        .parse(input)
}

fn parse_expression(input: &str) -> IResult<&str, ToyExpression> {
    alt((parse_expression_concat, parse_expression_atom)).parse(input)
}

fn parse_expression_concat(input: &str) -> IResult<&str, ToyExpression> {
    parse_expression_atom
        .and(parse_expression)
        .map(|(e1, e2)| ToyExpression::Concat {
            left: Box::new(e1),
            right: Box::new(e2),
        })
        .parse(input)
}

fn parse_expression_atom(input: &str) -> IResult<&str, ToyExpression> {
    alt((
        parse_prim.map(ToyExpression::Prim),
        parse_constant.map(ToyExpression::Constant),
        parse_var.map(ToyExpression::Var),
    ))
    .parse(input)
}

fn parse_prim(input: &str) -> IResult<&str, ToyPrim> {
    ws(alt((
        value(ToyPrim::Dup, tag("dup")),
        value(ToyPrim::Drop, tag("drop")),
        value(ToyPrim::Swap, tag("swap")),
        value(ToyPrim::Rot, tag("rot")),
        value(ToyPrim::Over, tag("over")),

        value(ToyPrim::Get, tag("get")),
        value(ToyPrim::Put, tag("put")),
        
        value(ToyPrim::Add, tag("add")),
        value(ToyPrim::Sub, tag("sub")),
        value(ToyPrim::Mul, tag("mul")),
        value(ToyPrim::Div, tag("div")),
        value(ToyPrim::Neg, tag("neg")),
        
        value(ToyPrim::Eq, tag("eq")),
        value(ToyPrim::Less, tag("less")),
        value(ToyPrim::And, tag("and")),
        value(ToyPrim::Or, tag("or")),
        value(ToyPrim::Not, tag("not")),
    )))
    .parse(input)
}

fn parse_var(input: &str) -> nom::IResult<&str, ToyVar> {
    ws(identifier)
        .map(|arg0: &str| ToyVar(arg0.to_owned()))
        .parse(input)
}

fn parse_constant(input: &str) -> nom::IResult<&str, ToyConstant> {
    ws(decimal)
        .map(|arg0: &str| ToyConstant(arg0.parse().unwrap()))
        .parse(input)
}

fn decimal(input: &str) -> IResult<&str, &str> {
    recognize(many1(terminated(one_of("0123456789"), many0(char('_'))))).parse(input)
}

fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ))
    .parse(input)
}

fn ws<'a, O, E: ParseError<&'a str>, F: Parser<&'a str, Output = O, Error = E>>(
    f: F,
) -> impl Parser<&'a str, Output = O, Error = E> {
    (multispace0, f).map(|(_, x)| x)
}
