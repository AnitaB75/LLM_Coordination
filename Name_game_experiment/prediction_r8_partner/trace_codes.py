"""Transparent trace-coding rules shared by the play and prediction analyses."""
import re
ID_STRICT = re.compile(r"(same model|instances? of (me|myself|yourself|opus|claude)|identical(ly)? (reason|copies|instances|players|agents|minds|logic)|"
                       r"(we|we're|we are|all) (all )?(reason|reasoning|think|thinking) (identically|the same)|reason identically|"
                       r"identical reasoners|copies of (me|myself)|same (reasoning|logic|thought process) as me|all (the )?same (model|agent))", re.I)
ID_HEDGED = re.compile(r"(everyone|others|other players|they|players|all of us) (might|may|would|will|could|likely)? ?(reason|think) (the )?same|"
                       r"same (way|reasoning|logic)", re.I)
TIE = re.compile(r"(alphabetic|(comes|listed|mentioned|appears) first|first (listed|mentioned|option|alphabetically)|canonical|deterministic|"
                 r"earlier in the alphabet|order (in which|they('re| are) (listed|presented))|listed order)", re.I)
NEG = re.compile(r"(doesn'?t|does not|no clear|not clearly|isn'?t|nothing)", re.I)
EMP = re.compile(r"(others?|everyone( else)?|other players|players|people|they|most|the group|player 2|the other (model|player))\b[^.]{0,40}?"
                 r"\b(will|would|might|may|are likely to|probably|likely|tend to|could)\b[^.]{0,25}?"
                 r"\b(choose|pick|switch|stay|stick|go|follow|defect|move|select|converge|abandon)", re.I)
HIST = re.compile(r"(last round|previous round|prior round|round \d+ results|player 2 (chose|picked|went|selected)|"
                  r"(most|majority)[^.]{0,20}(chose|picked|went|voted)|momentum|precedent|already (chose|picked|went|shifted)|votes?)", re.I)
UNC = re.compile(r"(uncertain|unpredictab|either way|coin ?flip|50/50|genuinely ambiguous|hard to (say|predict|know)|can'?t (know|be sure|tell)|no way to know|arbitrary)", re.I)
POLICY = re.compile(r"(\b(tend|tends) to\b|typically|usually|\boften\b|in my experience|my (past|previous|earlier) (choices|picks|behavio)|models (like|prefer|favor))", re.I)


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\s+[—-]{1,2}\s+|;\s*", t or "") if s.strip()]


def tie_present(t):
    return any(TIE.search(s) and not NEG.search(s) for s in sentences(t))


def code(t):
    t = t or ""; ss = sentences(t)
    idS = bool(ID_STRICT.search(t)); idH = bool(ID_HEDGED.search(t)) and not idS
    tie = tie_present(t)
    link = any((ID_STRICT.search(s) or ID_HEDGED.search(s)) and TIE.search(s) and not NEG.search(s) for s in ss) or any(
        (ID_STRICT.search(a) or ID_HEDGED.search(a)) and TIE.search(b) and not NEG.search(b)
        and re.search(r"\b(so|therefore|thus|hence|that's why|which means)\b", b, re.I) for a, b in zip(ss, ss[1:]))
    emp = any(EMP.search(s) and not ID_STRICT.search(s) for s in ss)
    return dict(identical=int(idS), identical_hedged=int(idH), tiebreak=int(tie), link=int(link and (idS or idH) and tie),
                empirical=int(emp), history=int(bool(HIST.search(t))), uncertainty=int(bool(UNC.search(t))),
                learned_policy=int(bool(POLICY.search(t))))
