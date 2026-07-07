"""MATH / MATH-500 grading aligned with EvalScope / Qwen2.5-Math.

Ports the core ``extract_answer`` + ``math_equal`` logic from EvalScope's
``math_parser.py`` (Qwen2.5-Math lineage) for sympy-aware equivalence checks.
"""
from __future__ import annotations

import re
from math import isclose

try:
    import regex
except ImportError:  # pragma: no cover
    import re as regex  # type: ignore

try:
    from word2number import w2n
except ImportError:  # pragma: no cover

    def _word_to_num(text: str) -> str:
        return text

else:

    def _word_to_num(text: str) -> str:
        try:
            return str(w2n.word_to_num(text))
        except Exception:
            return text


def convert_word_number(text: str) -> str:
    return _word_to_num(text)


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    a, b = parts[0], parts[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        if string == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
    except Exception:
        pass
    return string


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_answer_string(string: str) -> str:
    string = str(string).strip().replace("\n", "").rstrip(".")
    string = string.replace("\\!", "")
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")
    string = re.sub(
        r"\\text\{([a-zA-Z]+)\}",
        lambda m: convert_word_number(m.group(1).lower()),
        string,
    )
    string = re.sub(r"(cm|inches)\}\^2", r"\1}", string)
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string:
        string = _string
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    string = convert_word_number(string)
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ("x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"):
        string = string.replace(key, "")
    string = string.replace("\\emptyset", "{}").replace("(-\\infty,\\infty)", "\\mathbb{R}")
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if (
        (string.startswith("{") and string.endswith("}") and string[1:-1].replace(".", "").isalnum())
        or (string.startswith("(") and string.endswith(")"))
        or (string.startswith("[") and string.endswith("]"))
    ):
        string = string[1:-1]
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("and", "").replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)
    string = string.replace("'", "").replace('"', "")
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0].strip()) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string).replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    string = re.sub(r"\\(?=\-?\d+(\\|\)|,|\]|$))", "", string)
    string = re.sub(r"thgrade$", "", string)
    if re.fullmatch(r"\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*", string):
        string = string.replace(",", "")
    if re.fullmatch(r"(\s*-?\d+\s*,)*\s*-?\d+\s*", string):
        try:
            nums = sorted(int(x.strip()) for x in string.split(","))
            string = ",".join(str(x) for x in nums)
        except Exception:
            pass
    return string


_THINKING_BLOCK_RE = re.compile(
    r"<\s*redacted_thinking\s*>.*?<\s*/\s*redacted_thinking\s*>",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking_content(text: str) -> str:
    """Remove Qwen3 `` blocks before answer extraction."""
    return _THINKING_BLOCK_RE.sub("", text or "").strip()


def extract_math_answer(pred_str: str, use_last_number: bool = True) -> str:
    """Extract final answer from model completion (EvalScope-style)."""
    pred_str = strip_thinking_content(pred_str or "").replace("\u043a\u0438", "")
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        pred = pred_str.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if not ans:
            pred = ""
        elif ans[0] == "{":
            stack = 1
            parts: list[str] = []
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    parts.append(c)
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    parts.append(c)
                else:
                    parts.append(c)
            pred = "".join(parts)
        else:
            pred = ans.split("$")[0].strip()
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    elif "ANSWER:" in pred_str:
        pred = pred_str.split("ANSWER:")[-1].strip()
    elif use_last_number:
        nums = re.findall(r"-?\d*\.?\d+", pred_str.replace(",", ""))
        pred = nums[-1] if nums else ""
    else:
        pred = ""
    pred = re.sub(r"\n\s*", "", pred)
    if pred.startswith(":"):
        pred = pred[1:]
    if pred.endswith((".", "/")):
        pred = pred[:-1]
    return strip_answer_string(pred)


def choice_answer_clean(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp[-1]
    pred = str(pred).rstrip(".").rstrip("/")
    return pred


def parse_digits(num: str):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            try:
                return float(num[:-1]) / 100
            except Exception:
                pass
        if num.endswith("\\"):
            try:
                return float(num[:-1])
            except Exception:
                pass
    return None


def is_digit(num: str) -> bool:
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def str_to_pmatrix(input_str: str) -> str:
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for m in matrix_str:
        m = m.strip("{}")
        pmatrix_list.append(r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}")
    return ", ".join(pmatrix_list)


def symbolic_equal(a, b) -> bool:
    from sympy import N, simplify

    parse_fns = []
    try:
        from sympy.parsing.latex import parse_latex

        parse_fns.append(parse_latex)
    except Exception:
        pass
    try:
        from sympy.parsing.sympy_parser import parse_expr

        parse_fns.append(parse_expr)
    except Exception:
        pass
    try:
        from latex2sympy2_extended import latex2sympy

        parse_fns.append(latex2sympy)
    except Exception:
        pass

    def _parse(s):
        for f in parse_fns:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass
    return False


def math_equal(
    prediction: str,
    reference: str,
    *,
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    if prediction is None or reference is None:
        return False
    prediction = strip_answer_string(str(prediction))
    reference = strip_answer_string(str(reference))
    if not prediction and prediction not in ("0",):
        return False
    if prediction.lower() == reference.lower():
        return True
    if reference in ("A", "B", "C", "D", "E") and choice_answer_clean(prediction) == reference:
        return True

    try:
        if is_digit(prediction) and is_digit(reference):
            pred_f = parse_digits(prediction)
            ref_f = parse_digits(reference)
            gt_result = [ref_f / 100, ref_f, ref_f * 100] if include_percentage else [ref_f]
            for item in gt_result:
                try:
                    if is_close and numeric_equal(pred_f, item):
                        return True
                    if not is_close and item == pred_f:
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    pred_str, ref_str = prediction, reference
    if (
        (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("("))
        or (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("["))
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
        for ch in ("{", "}", "(", ")"):
            ref_str = ref_str.replace(ch, "")
            pred_str = pred_str.replace(ch, "")
        if pred_str.lower() == ref_str.lower():
            return True

    if regex.match(r"(\(|\[).+(\)|\])", prediction) and regex.match(r"(\(|\[).+(\)|\])", reference):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(pred_parts[i], ref_parts[i], include_percentage=include_percentage, is_close=is_close)
            for i in range(len(pred_parts))
        ):
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        p_lhs, p_rhs = prediction.split("=", 1)
        r_lhs, r_rhs = reference.split("=", 1)
        pred_expr = f"{p_lhs.strip()} - ({p_rhs.strip()})"
        ref_expr = f"{r_lhs.strip()} - ({r_rhs.strip()})"
        if symbolic_equal(pred_expr, ref_expr) or symbolic_equal(f"-({pred_expr})", ref_expr):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=", 1)[1], reference, include_percentage=include_percentage, is_close=is_close):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(prediction, reference.split("=", 1)[1], include_percentage=include_percentage, is_close=is_close):
            return True

    return symbolic_equal(prediction, reference)
