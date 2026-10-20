"""Tier-1 variants (test-set freeze stage 2, 2026-10-19). Written before any model has seen the sources or variants.

Three variants per primary source in ../stage1_run1/files/tier1_variant_sources.jsonl:
  wording   - same mathematics and answer, reworded;
  numbers   - same structure, new numbers, new answer;
  condition - a condition or the asked quantity changes, new answer.
Every variant carries `check`, an independent computation (brute force, sympy or numerics) of its answer, and
`value`, the answer as a sympy expression string (or a plain string for text / base-b answers). check_variants.py
runs every check, compares it with `value`, and checks that the LaTeX `answer` denotes the same value.
"""
import cmath
import itertools
import math
from fractions import Fraction

import sympy as sp

x, y, z, n_ = sp.symbols('x y z n', real=True)


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def rev(n):
    return int(str(n)[::-1])


def reversible_primes(lo, hi):
    return [p for p in range(lo, hi) if is_prime(p) and is_prime(rev(p))]


def overlap_min_max(n, a, b):
    """Smallest and largest |A ∩ B| over all ways to choose |A| = a, |B| = b in an n-set (brute force over counts)."""
    both = [k for k in range(0, min(a, b) + 1) if a + b - k <= n]
    return min(both), max(both)


def prism_counts(k):
    """Vertices, edges, faces of a k-gonal prism, counted from its construction; Euler's formula is asserted."""
    v = 2 * k
    e = k + k + k  # two k-gon rims and k lateral edges
    f = 2 + k
    assert v - e + f == 2
    return v, e, f


def weekday(start, offset):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return days[(days.index(start) + offset) % 7]


def clock_angle(h1, h2):
    d = abs(h1 - h2) % 12 * 30
    return min(d, 360 - d)


def square_other_vertices(p, q):
    cx, cy = (Fraction(p[0] + q[0], 2), Fraction(p[1] + q[1], 2))
    vx, vy = p[0] - cx, p[1] - cy
    return (cx - vy, cy + vx), (cx + vy, cy - vx)


def max_gcd(total, distinct=False):
    return max(math.gcd(a, total - a) for a in range(1, total) if not (distinct and 2 * a == total))


def round_half_up(v, unit):
    return (v + unit // 2) // unit * unit


def letters_rounding(values, target, unit=10):
    return ''.join(ch for ch, v in zip('ABCDEF', values) if round_half_up(v, unit) == target)


def cows(d, price=lambda k: k, rng=range(1, 3000)):
    """Amounts the younger brother can get at the last division: pay d alternately starting with the older brother,
    the older brother receives the last full d, the younger then gets 0 < r < d."""
    out = set()
    for k in rng:
        m = k * price(k) % (2 * d)
        if d < m < 2 * d:
            out.add(m - d)
    return out


def cows_difference(d=10, rng=range(1, 3000)):
    out = set()
    for k in rng:
        total, older, younger, turn = k * k, 0, 0, 0
        if not (d < total % (2 * d) < 2 * d):
            continue
        while total > 0:
            pay = min(d, total)
            if turn == 0:
                older += pay
            else:
                younger += pay
            total -= pay
            turn ^= 1
        out.add(older - younger)
    return out


def to_base(n, b):
    s = ''
    while n:
        s = str(n % b) + s
        n //= b
    return s or '0'


def round_to(v, unit):
    return round_half_up(v, unit)


def rounding_gap(nums, unit=10):
    return round_to(sum(nums), unit) - sum(round_to(v, unit) for v in nums)


def numeric_min(f, dims, trials=4000, seed=0):
    import random
    rnd = random.Random(seed)
    best = math.inf
    for _ in range(trials):
        best = min(best, f(*[math.exp(rnd.uniform(-3, 3)) for _ in range(dims)]))
    return best


def ratio_628(k):
    """Sum of reciprocals of k numbers over the sum of reciprocals of their pairwise sums."""
    return lambda *t: sum(1 / u for u in t) / sum(1 / (p + q) for p, q in itertools.combinations(t, 2))


def attained_bound(f, dims, bound, sense, trials=20000):
    """`bound` is the min (or max) of f over positive reals: f at all-equal arguments equals bound exactly, and no
    random log-uniform sample beats it."""
    assert f(*[Fraction(1)] * dims) == bound
    sign = 1 if sense == 'min' else -1
    assert numeric_min(lambda *t: sign * (f(*t) - float(bound)), dims, trials) >= -1e-12
    return bound


def sphere_max(fn, radius2=1.0, steps=181):
    """Maximum of fn over positive x, y, z with x^2 + y^2 + z^2 = radius2 (grid over spherical angles)."""
    r = math.sqrt(radius2)
    best = -math.inf
    for i in range(1, steps):
        th = math.pi / 2 * i / steps
        for j in range(1, steps):
            ph = math.pi / 2 * j / steps
            best = max(best, fn(r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)))
    return best


def domain_of(expr):
    from sympy.calculus.util import continuous_domain
    return continuous_domain(expr, x, sp.S.Reals)


def solve_a(shift, mult, at):
    """All a != 0 with f(at) = f^{-1}(mult * a) for f(x) = a / (x + shift), checked by direct substitution."""
    a = sp.symbols('a')
    f = lambda t: a / (t + shift)
    sols = sp.solve(sp.Eq(f(f(at)), mult * a), a)
    good = []
    for s in sols:
        if s == 0:
            continue  # f is constant 0, no inverse
        fv = f(at).subs(a, s)
        if (fv + shift) == 0:
            continue
        assert sp.simplify(f(fv).subs(a, s) - mult * s) == 0
        good.append(s)
    assert len(good) == 1, good
    return good[0]


def recurrence_pqr(a1, c, k):
    """a_1 = a1, a_n = c + k a_{n-1}; returns p, q, r with a_n = p q^n + r, verified on the first 12 terms."""
    r = sp.Rational(c, 1 - k)
    q = k
    p = sp.Rational(a1 - r, q)
    seq = [a1]
    for _ in range(11):
        seq.append(c + k * seq[-1])
    assert all(p * q ** (i + 1) + r == v for i, v in enumerate(seq))
    return p, q, r


def unity_sum(n, g):
    rs = [cmath.exp(2j * cmath.pi * k / (n + 1)) for k in range(1, n + 1)]
    s = sum(g(r) for r in rs)
    assert abs(s.imag) < 1e-9
    return round(s.real, 9)


def cubic_diag(coeffs):
    roots = sp.Poly(coeffs, x).nroots()
    assert all(abs(sp.im(r)) < 1e-12 and sp.re(r) > 0 for r in roots), roots
    return math.sqrt(sum(float(sp.re(r)) ** 2 for r in roots)), roots


def v(family, kind, problem, answer, value, check):
    return {'family': family, 'type': kind, 'problem': problem, 'answer': answer, 'value': value, 'check': check}


VARIANTS = [
    # 1 prealgebra/test/134
    v('prealgebra/test/134', 'wording', 'What is the quotient when negative forty-nine is divided by seven?',
      '-7', '-7', lambda: Fraction(-49, 7)),
    v('prealgebra/test/134', 'numbers', 'Compute $(-96)\\div 8$.', '-12', '-12', lambda: Fraction(-96, 8)),
    v('prealgebra/test/134', 'condition', 'Compute $(-49)\\div (-7)$.', '7', '7', lambda: Fraction(-49, -7)),
    # 2 geometry/test/347
    v('geometry/test/347', 'wording', 'A solid has two parallel faces that are congruent octagons, and its remaining '
      'faces are rectangles joining corresponding sides of the two octagons. How many edges does this solid have?',
      '24', '24', lambda: prism_counts(8)[1]),
    v('geometry/test/347', 'numbers', 'How many edges does a decagonal prism have?', '30', '30',
      lambda: prism_counts(10)[1]),
    v('geometry/test/347', 'condition', 'How many faces does an octagonal prism have?', '10', '10',
      lambda: prism_counts(8)[2]),
    # 3 counting_and_probability/test/10
    v('counting_and_probability/test/10', 'wording', "Mr. Shearer teaches a class of 28 students. Exactly three "
      "quarters of them have brown hair, and exactly six sevenths of them are right-handed. What is the least number "
      "of students in the class who could be both brown-haired and right-handed?", '17', '17',
      lambda: overlap_min_max(28, 21, 24)[0]),
    v('counting_and_probability/test/10', 'numbers', "Two-thirds of the students in Ms. Alvarez's class have brown "
      "eyes and five-sixths of her students are right-handed. If Ms. Alvarez's class has 36 students, what is the "
      "smallest possible number of students that could be both right-handed and have brown eyes?", '18', '18',
      lambda: overlap_min_max(36, 24, 30)[0]),
    v('counting_and_probability/test/10', 'condition', "Three-fourths of the students in Mr. Shearer's class have "
      "brown hair and six-sevenths of his students are right-handed. If Mr. Shearer's class has 28 students, what is "
      "the largest possible number of students that could be both right-handed and have brown hair?", '21', '21',
      lambda: overlap_min_max(28, 21, 24)[1]),
    # 4 number_theory/test/290
    v('number_theory/test/290', 'wording', 'Call a prime "reversible" if writing its decimal digits in the opposite '
      'order also gives a prime. Find the least reversible prime that exceeds 17.', '31', '31',
      lambda: reversible_primes(18, 1000)[0]),
    v('number_theory/test/290', 'numbers', "Let's call a prime number a $\\textit{reversible prime}$ if the number "
      "formed by reversing its digits is also prime. What is the smallest reversible prime greater than 40?",
      '71', '71', lambda: reversible_primes(41, 1000)[0]),
    v('number_theory/test/290', 'condition', "Let's call a prime number a $\\textit{reversible prime}$ if the number "
      "formed by reversing its digits is also prime. What is the largest two-digit reversible prime?", '97', '97',
      lambda: reversible_primes(10, 100)[-1]),
    # 5 number_theory/test/340
    v('number_theory/test/340', 'wording', 'Find the smallest prime $p$ with $p > 25$ such that dividing $p$ by $25$ '
      'leaves a remainder of $2$.', '127', '127', lambda: next(p for p in range(26, 10 ** 4) if is_prime(p) and p % 25 == 2)),
    v('number_theory/test/340', 'numbers', 'What is the least prime number greater than 24 that will have a remainder '
      'of 1 when divided by 24?', '73', '73', lambda: next(p for p in range(25, 10 ** 4) if is_prime(p) and p % 24 == 1)),
    v('number_theory/test/340', 'condition', 'What is the second smallest prime number greater than 25 that will have '
      'a remainder of 2 when divided by 25?', '227', '227',
      lambda: [p for p in range(26, 10 ** 4) if is_prime(p) and p % 25 == 2][1]),
    # 6 algebra/test/767
    v('algebra/test/767', 'wording', 'Multiply $6^2 - 3\\cdot 11$ by $8$, divide the product by $8$, and then add $3$. '
      'What number results?', '6', '6',
      lambda: 3 + Fraction(8 * (6 ** 2 - 3 * 11), 8)),
    v('algebra/test/767', 'numbers', 'Find the value of $6[5^2-2(9)]\\div3 + 4$.', '18', '18',
      lambda: Fraction(6 * (5 ** 2 - 2 * 9), 3) + 4),
    v('algebra/test/767', 'condition', 'Find the value of $8[6^2-3(11)]\\div(8 + 4)$.', '2', '2',
      lambda: Fraction(8 * (6 ** 2 - 3 * 11), 8 + 4)),
    # 7 algebra/test/814
    v('algebra/test/814', 'wording', 'The same number is added to both the numerator and the denominator of '
      '$\\frac{5}{8}$, and the resulting fraction equals $\\frac{2}{5}$. What is that number?', '-3', '-3',
      lambda: sp.solve(sp.Eq((5 + x) / (8 + x), sp.Rational(2, 5)), x)[0]),
    v('algebra/test/814', 'numbers', 'What number, when added to the numerator and to the denominator of '
      '$\\frac{3}{7}$, results in a fraction whose value is 0.6?', '3', '3',
      lambda: sp.solve(sp.Eq((3 + x) / (7 + x), sp.Rational(3, 5)), x)[0]),
    v('algebra/test/814', 'condition', 'What number, when added to the numerator of $\\frac{5}{8}$ and subtracted from '
      'its denominator, results in a fraction whose value is 0.4?', '-\\frac{9}{7}', '-9/7',
      lambda: sp.solve(sp.Eq((5 + x) / (8 - x), sp.Rational(2, 5)), x)[0]),
    # 8 prealgebra/test/356
    v('prealgebra/test/356', 'wording', 'Angle $M$ and an angle measuring 10 degrees together form a right angle. '
      'How many degrees are in angle $M$?', '80', '80', lambda: 90 - 10),
    v('prealgebra/test/356', 'numbers', 'The complement of angle $M$ is 37 degrees. What is the measure, in degrees, '
      'of angle $M$?', '53', '53', lambda: 90 - 37),
    v('prealgebra/test/356', 'condition', 'The supplement of angle $M$ is 10 degrees. What is the measure, in degrees, '
      'of angle $M$?', '170', '170', lambda: 180 - 10),
    # 9 algebra/test/1014
    v('algebra/test/1014', 'wording', 'Let $a = 1$, $b = 3$, and $c = 5$. Find the value of $a^3 + b^2 + c$ divided '
      'by $a + b^2 + c^3$.', '\\frac{1}{9}', '1/9', lambda: Fraction(1 + 9 + 5, 1 + 9 + 125)),
    v('algebra/test/1014', 'numbers', 'Evaluate $\\displaystyle\\frac{a^3 + b^2 + c}{a + b^2 + c^3}$ when $a = 2$, '
      '$b = 1$, and $c = 3$.', '\\frac{2}{5}', '2/5', lambda: Fraction(2 ** 3 + 1 + 3, 2 + 1 + 3 ** 3)),
    v('algebra/test/1014', 'condition', 'Evaluate $\\displaystyle\\frac{a^3 + b^2 + c}{a + b^2 + c^3}$ when $a = 1$, '
      '$b = 3$, and $c = -5$.', '-\\frac{1}{23}', '-1/23', lambda: Fraction(1 + 9 - 5, 1 + 9 + (-5) ** 3)),
    # 10 geometry/test/82
    v('geometry/test/82', 'wording', 'A ball used for hand exercises is a sphere measuring 2 inches across through its '
      'center. What is its surface area, in square inches? Express your answer in terms of $\\pi$.', '4\\pi',
      '4*pi', lambda: 4 * sp.pi * sp.Rational(2, 2) ** 2),
    v('geometry/test/82', 'numbers', "In China, spheres are used to exercise a person's hands. If the diameter of a "
      "sphere is 6 inches, how many square inches are in its surface area? Express your answer in terms of $\\pi$.",
      '36\\pi', '36*pi', lambda: 4 * sp.pi * sp.Rational(6, 2) ** 2),
    v('geometry/test/82', 'condition', "In China, spheres are used to exercise a person's hands. If the diameter of a "
      "sphere is 2 inches, how many cubic inches are in its volume? Express your answer in terms of $\\pi$.",
      '\\frac{4\\pi}{3}', '4*pi/3', lambda: sp.Rational(4, 3) * sp.pi * sp.Rational(2, 2) ** 3),
    # 11 number_theory/test/328
    v('number_theory/test/328', 'wording', 'Today is a Monday. On which day of the week will it be exactly 270 days '
      'from today?', '\\text{Friday}', 'Friday', lambda: weekday('Monday', 270)),
    v('number_theory/test/328', 'numbers', 'If March 1 is a Wednesday, what day of the week will it be 200 days later?',
      '\\text{Sunday}', 'Sunday', lambda: weekday('Wednesday', 200)),
    v('number_theory/test/328', 'condition', 'If March 1 is a Monday, what day of the week was it 270 days earlier?',
      '\\text{Thursday}', 'Thursday', lambda: weekday('Monday', -270)),
    # 12 geometry/test/168
    v('geometry/test/168', 'wording', 'Think of a steering wheel as a clock face. Drivers are taught to hold the wheel '
      'at the "$3$" and the "$9$". If a driver holds it at the "$2$" and the "$10$" instead, by how many degrees does '
      'the angle between the two hands, measured at the center of the wheel, get smaller?', '60^\\circ', '60',
      lambda: clock_angle(3, 9) - clock_angle(2, 10)),
    v('geometry/test/168', 'numbers', 'At a driver education class, the students were told to put their hands on the '
      'steering wheel on "$3$" and "$9$", just as on the face of a clock. If, instead, the students placed their hands '
      'on the "$1$" and the "$11$", by how many degrees would the angle between the hands (with the center of the '
      'wheel at the vertex of the angle) decrease?', '120^\\circ', '120',
      lambda: clock_angle(3, 9) - clock_angle(1, 11)),
    v('geometry/test/168', 'condition', 'At a driver education class, the students were told to put their hands on the '
      'steering wheel on "$3$" and "$9$", just as on the face of a clock. If, instead, a student placed the hands on '
      'the "$2$" and the "$9$", what would be the measure, in degrees, of the smaller angle between the hands (with the '
      'center of the wheel at the vertex of the angle)?', '150^\\circ', '150', lambda: clock_angle(2, 9)),
    # 13 algebra/test/158
    v('algebra/test/158', 'wording', 'A rectangle has a perimeter of 12 inches. What is the largest area, in square '
      'inches, that it can enclose?', '9', '9',
      lambda: sp.maximum(x * (6 - x), x, sp.Interval.open(0, 6))),
    v('algebra/test/158', 'numbers', 'What is the maximum number of square inches in the area of a rectangle with a '
      'perimeter of 30 inches?', '\\frac{225}{4}', '225/4',
      lambda: sp.maximum(x * (15 - x), x, sp.Interval.open(0, 15))),
    v('algebra/test/158', 'condition', 'A rectangle is placed against a straight wall, so that only three of its sides '
      'need fencing. If 12 inches of fencing are used for those three sides, what is the maximum number of square '
      'inches in the area of the rectangle?', '18', '18',
      lambda: sp.maximum(x * (12 - 2 * x), x, sp.Interval.open(0, 6))),
    # 14 algebra/test/796
    v('algebra/test/796', 'wording', 'A square has opposite corners at $(7,9)$ and $(10,2)$. Add the $y$-coordinates of '
      'its other two corners. What is the result?', '11', '11',
      lambda: sum(p[1] for p in square_other_vertices((7, 9), (10, 2)))),
    v('algebra/test/796', 'numbers', 'If $(1,4)$ and $(5,10)$ are the coordinates of two opposite vertices of a square, '
      'what is the sum of the $y$-coordinates of the other two vertices?', '14', '14',
      lambda: sum(p[1] for p in square_other_vertices((1, 4), (5, 10)))),
    v('algebra/test/796', 'condition', 'If $(7,9)$ and $(10,2)$ are the coordinates of two opposite vertices of a '
      'square, what is the area of the square?', '29', '29',
      lambda: (lambda o: (o[0][0] - 7) ** 2 + (o[0][1] - 9) ** 2)(square_other_vertices((7, 9), (10, 2)))),
    # 15 number_theory/test/520
    v('number_theory/test/520', 'wording', 'Two positive integers add up to 80. What is the greatest value that their '
      'greatest common divisor can take?', '40', '40', lambda: max_gcd(80)),
    v('number_theory/test/520', 'numbers', 'The sum of two positive integers $a$ and $b$ is 91.  What is the largest '
      'possible value of $\\gcd(a,b)$?', '13', '13', lambda: max_gcd(91)),
    v('number_theory/test/520', 'condition', 'The sum of two distinct positive integers $a$ and $b$ is 80.  What is the '
      'largest possible value of $\\gcd(a,b)$?', '20', '20', lambda: max_gcd(80, distinct=True)),
    # 16 prealgebra/test/296
    v('prealgebra/test/296', 'wording', 'Six integers are listed below.\nA. $43$\nB. $23$\nC. $38$\nD. $51$\nE. $45$\n'
      'F. $35$\nWhich of them become $40$ after rounding to the nearest ten? Give the letters of those integers as one '
      'block in alphabetical order with no punctuation (for instance, if every integer qualified, the answer would be '
      'ABCDEF).', '\\text{ACF}', 'ACF', lambda: letters_rounding([43, 23, 38, 51, 45, 35], 40)),
    v('prealgebra/test/296', 'numbers', 'Which of the following integers round to $70$ when rounded to the nearest ten?\n'
      'A. $74$\nB. $65$\nC. $76$\nD. $68$\nE. $75$\nF. $81$\nExpress your answer as a block of letters corresponding to '
      'the answer choices that round to $70$, in alphabetic order with no punctuation. For example, if all six answer '
      'choices rounded to $70$, then your answer would be ABCDEF.', '\\text{ABD}', 'ABD',
      lambda: letters_rounding([74, 65, 76, 68, 75, 81], 70)),
    v('prealgebra/test/296', 'condition', 'Which of the following integers round to $50$ when rounded to the nearest '
      'ten?\nA. $43$\nB. $23$\nC. $38$\nD. $51$\nE. $45$\nF. $35$\nExpress your answer as a block of letters '
      'corresponding to the answer choices that round to $50$, in alphabetic order with no punctuation. For example, if '
      'all six answer choices rounded to $50$, then your answer would be ABCDEF.', '\\text{DE}', 'DE',
      lambda: letters_rounding([43, 23, 38, 51, 45, 35], 50)),
    # 17 intermediate_algebra/test/628
    v('intermediate_algebra/test/628', 'wording', 'For positive real numbers $a,$ $b,$ $c,$ consider the sum of their '
      'reciprocals divided by the sum of the reciprocals of $a + b,$ $a + c,$ and $b + c.$  What is the smallest value '
      'this quotient can take?', '2', '2', lambda: attained_bound(ratio_628(3), 3, Fraction(2), 'min')),
    v('intermediate_algebra/test/628', 'numbers', 'Four quantities $a,$ $b,$ $c,$ $d$ are all greater than zero.  Over '
      'all such choices, what is the least possible value of\n\\[\\frac{\\frac{1}{a} + \\frac{1}{b} + \\frac{1}{c} + \\frac{1}{d}}{\\frac{1}{a + b} + \\frac{1}{a + c} '
      '+ \\frac{1}{a + d} + \\frac{1}{b + c} + \\frac{1}{b + d} + \\frac{1}{c + d}}.\\]', '\\frac{4}{3}', '4/3',
      lambda: attained_bound(ratio_628(4), 4, Fraction(4, 3), 'min')),
    v('intermediate_algebra/test/628', 'condition', 'Let $a,$ $b,$ $c$ be positive real numbers.  Find the maximum value '
      'of\n\\[\\frac{\\frac{1}{a + b} + \\frac{1}{a + c} + \\frac{1}{b + c}}{\\frac{1}{a} + \\frac{1}{b} + \\frac{1}{c}}.\\]',
      '\\frac{1}{2}', '1/2', lambda: attained_bound(lambda *t: 1 / ratio_628(3)(*t), 3, Fraction(1, 2), 'max')),
    # 18 prealgebra/test/736
    v('prealgebra/test/736', 'wording', 'At a certain moment, a 5-foot-tall woman casts a shadow 8 feet long, and a '
      'nearby tree casts a shadow 120 feet long. How many feet tall is the tree?', '75', '75',
      lambda: Fraction(5, 8) * 120),
    v('prealgebra/test/736', 'numbers', 'A man 6-feet tall casts a 4-foot shadow. At the same time of day, the shadow of '
      'a tree nearby is 42 feet long. In feet, how tall is the tree?', '63', '63', lambda: Fraction(6, 4) * 42),
    v('prealgebra/test/736', 'condition', 'A woman 5-feet tall casts an 8-foot shadow. At the same time of day, a tree '
      'nearby is 120 feet tall. In feet, how long is the shadow of the tree?', '192', '192',
      lambda: Fraction(8, 5) * 120),
    # 19 number_theory/test/169
    v('number_theory/test/169', 'wording', 'Two brothers brought $x$ cows to market and sold every cow for $x$ dollars. '
      'They split the proceeds by handing out $\\$10$ to the older brother, then $\\$10$ to the younger, then $\\$10$ to '
      'the older, and so on, alternating. The final full $\\$10$ went to the older brother, after which the younger '
      'brother received the remaining amount, which was less than $\\$10$. How many dollars did the younger brother '
      'receive in that last division?', '6', '6', lambda: (lambda s: s.pop() if len(s) == 1 else None)(cows(10))),
    v('number_theory/test/169', 'numbers', 'Two brothers took $x$ cows to the market, and they sold each cow for $x$ '
      'dollars. After selling the cows, they divided the money by giving $\\$12$ to the older brother, $\\$12$ to the '
      'younger brother, $\\$12$ to the older, $\\$12$ to the younger, and so on. At the last division of money, the '
      'older brother received $\\$12$, but the younger brother received less than $\\$12$. How many dollars must the '
      'younger brother have received at the last division?', '4', '4',
      lambda: (lambda s: s.pop() if len(s) == 1 else None)(cows(12))),
    v('number_theory/test/169', 'condition', 'Two brothers took $x$ cows to the market, and they sold each cow for $x$ '
      'dollars. After selling the cows, they divided the money by giving $\\$10$ to the older brother, $\\$10$ to the '
      'younger brother, $\\$10$ to the older, $\\$10$ to the younger, and so on. At the last division of money, the '
      'older brother received $\\$10$, but the younger brother received less than $\\$10$. In total, how many more '
      'dollars did the older brother receive than the younger brother?', '4', '4',
      lambda: (lambda s: s.pop() if len(s) == 1 else None)(cows_difference(10))),
    # 20 algebra/test/548
    v('algebra/test/548', 'wording', 'Let $f(x) = x^2-3$ and let $t(x) = 9+2f(x)$. Compute $t(f(-2))$.', '5', '5',
      lambda: (lambda f: (lambda t: t(f(-2)))(lambda u: 9 + 2 * f(u)))(lambda u: u ** 2 - 3)),
    v('algebra/test/548', 'numbers', 'If $t(x) = 7+3f(x)$ and $f(x) = x^2-5$, then what is $t(f(3))$?', '40', '40',
      lambda: (lambda f: (lambda t: t(f(3)))(lambda u: 7 + 3 * f(u)))(lambda u: u ** 2 - 5)),
    v('algebra/test/548', 'condition', 'If $t(x) = 9+2f(x)$ and $f(x) = x^2-3$, then what is $f(t(-2))$?', '118', '118',
      lambda: (lambda f: (lambda t: f(t(-2)))(lambda u: 9 + 2 * f(u)))(lambda u: u ** 2 - 3)),
    # 21 number_theory/test/1
    v('number_theory/test/1', 'wording', "A plumber's rate is $242_5$ dollars per hour of labor, plus a one-time "
      "equipment charge of $367_{8}$ dollars. The plumber works on Kirsty's house for $3.5_{10}$ hours. In base ten, "
      "how many dollars does Kirsty owe in total?", '499', '499',
      lambda: Fraction(7, 2) * int('242', 5) + int('367', 8)),
    v('number_theory/test/1', 'numbers', 'Kirsty needs to hire a plumber to fix her house. The plumber charges $134_5$ '
      'dollars for every hour of labor and $205_{8}$ dollars for equipment. If the plumber works for $4.5_{10}$ hours, '
      'how many dollars (in base ten) will Kirsty owe the plumber?', '331', '331',
      lambda: Fraction(9, 2) * int('134', 5) + int('205', 8)),
    v('number_theory/test/1', 'condition', 'Kirsty needs to hire a plumber to fix her house. The plumber charges $242_5$ '
      'dollars for every hour of labor and $367_{8}$ dollars for equipment. If Kirsty owes the plumber $607_{10}$ '
      'dollars in total, for how many hours did the plumber work?', '5', '5',
      lambda: Fraction(607 - int('367', 8), int('242', 5))),
    # 22 number_theory/test/174
    v('number_theory/test/174', 'wording', 'Subtract $212_3$ from $2121_3$. Write the difference in base 3.',
      '1202_3', '1202_3', lambda: to_base(int('2121', 3) - int('212', 3), 3) + '_3'),
    v('number_theory/test/174', 'numbers', 'What is $3102_4 - 233_4?$ Express your answer in base 4.', '2203_4',
      '2203_4', lambda: to_base(int('3102', 4) - int('233', 4), 4) + '_4'),
    v('number_theory/test/174', 'condition', 'What is $2121_3 + 212_3?$ Express your answer in base 3.', '10110_3',
      '10110_3', lambda: to_base(int('2121', 3) + int('212', 3), 3) + '_3'),
    # 23 prealgebra/test/247
    v('prealgebra/test/247', 'wording', 'Take the numbers 13, 14, 22, 52, 63, 74. Let $a$ be their sum rounded to the '
      'nearest ten, and let $b$ be the sum obtained after first rounding each of them to the nearest ten. Find $a-b$.',
      '20', '20', lambda: rounding_gap([13, 14, 22, 52, 63, 74])),
    v('prealgebra/test/247', 'numbers', 'We are given the following numbers: 16, 24, 35, 41, 58, 67. Calculate $a$ by '
      'adding the numbers and then rounding the sum to the nearest ten. Calculate $b$ by first rounding each number to '
      'the nearest ten and then adding the rounded numbers. What is $a-b$?', '-10', '-10',
      lambda: rounding_gap([16, 24, 35, 41, 58, 67])),
    v('prealgebra/test/247', 'condition', 'We are given the following numbers: 13, 14, 22, 52, 63, 74. Calculate $a$ by '
      'adding the numbers and then rounding the sum to the nearest hundred. Calculate $b$ by first rounding each number '
      'to the nearest hundred and then adding the rounded numbers. What is $a-b$?', '-100', '-100',
      lambda: rounding_gap([13, 14, 22, 52, 63, 74], 100)),
    # 24 intermediate_algebra/test/67
    v('intermediate_algebra/test/67', 'wording', 'The real numbers $r,$ $s,$ $t$ are the three roots of '
      '$x^3 - 4x^2 + 5x - \\frac{19}{10} = 0.$  A rectangular box has edge lengths $r,$ $s,$ and $t.$  How long is the '
      'segment joining two opposite corners of the box through its interior?', '\\sqrt{6}', 'sqrt(6)',
      lambda: sp.nsimplify(round(cubic_diag([1, -4, 5, sp.Rational(-19, 10)])[0] ** 2, 9)) ** sp.Rational(1, 2)),
    v('intermediate_algebra/test/67', 'numbers', 'The equation $x^3 - 6x^2 + 10x - 3 = 0$ has real roots $r,$ $s,$ and '
      '$t.$  Find the length of the long diagonal of a box with sides of length $r,$ $s,$ and $t.$', '4', '4',
      lambda: sp.nsimplify(round(cubic_diag([1, -6, 10, -3])[0] ** 2, 9)) ** sp.Rational(1, 2)),
    v('intermediate_algebra/test/67', 'condition', 'The equation $x^3 - 4x^2 + 5x - \\frac{19}{10} = 0$ has real roots '
      '$r,$ $s,$ and $t.$  Find the total surface area of a box with sides of length $r,$ $s,$ and $t.$', '10', '10',
      lambda: (lambda rs: sp.nsimplify(round(float(2 * (rs[0] * rs[1] + rs[0] * rs[2] + rs[1] * rs[2])), 9)))(
          [sp.re(r) for r in cubic_diag([1, -4, 5, sp.Rational(-19, 10)])[1]])),
    # 25 intermediate_algebra/test/15
    v('intermediate_algebra/test/15', 'wording', 'Positive real numbers $x,$ $y,$ $z$ satisfy $x^2 + y^2 + z^2 = 1.$  '
      'What is the largest possible value of $\\sqrt{10}\\,xy + yz$?', '\\frac{\\sqrt{11}}{2}', 'sqrt(11)/2',
      lambda: sp.sqrt(11) / 2 if abs(sphere_max(lambda a, b, c: math.sqrt(10) * a * b + b * c) - math.sqrt(11) / 2) < 1e-3 else None),
    v('intermediate_algebra/test/15', 'numbers', 'Let $x,$ $y,$ $z$ be positive real numbers such that '
      '$x^2 + y^2 + z^2 = 1.$  Find the maximum value of\n\\[xy \\sqrt{8} + yz.\\]', '\\frac{3}{2}', '3/2',
      lambda: sp.Rational(3, 2) if abs(sphere_max(lambda a, b, c: math.sqrt(8) * a * b + b * c) - 1.5) < 1e-3 else None),
    v('intermediate_algebra/test/15', 'condition', 'Let $x,$ $y,$ $z$ be positive real numbers such that '
      '$x^2 + y^2 + z^2 = 4.$  Find the maximum value of\n\\[xy \\sqrt{10} + yz.\\]', '2\\sqrt{11}', '2*sqrt(11)',
      lambda: 2 * sp.sqrt(11) if abs(sphere_max(lambda a, b, c: math.sqrt(10) * a * b + b * c, 4.0) - 2 * math.sqrt(11)) < 4e-3 else None),
    # 26 algebra/test/948
    v('algebra/test/948', 'wording', 'For which real numbers $x$ is $f(x)=\\frac{x+2}{x^2-2x-24}$ defined? Give the '
      'answer in interval notation.', '(-\\infty,-4)\\cup(-4,6)\\cup(6,\\infty)',
      'Union(Interval.open(-oo,-4),Interval.open(-4,6),Interval.open(6,oo))',
      lambda: domain_of((x + 2) / (x ** 2 - 2 * x - 24))),
    v('algebra/test/948', 'numbers', 'What is the domain of the function $f(x)=\\frac{x-1}{x^2+x-12}$? Express your '
      'answer in interval notation.', '(-\\infty,-4)\\cup(-4,3)\\cup(3,\\infty)',
      'Union(Interval.open(-oo,-4),Interval.open(-4,3),Interval.open(3,oo))',
      lambda: domain_of((x - 1) / (x ** 2 + x - 12))),
    v('algebra/test/948', 'condition', 'What is the domain of the function $f(x)=\\frac{x+2}{\\sqrt{x^2-2x-24}}$? '
      'Express your answer in interval notation.', '(-\\infty,-4)\\cup(6,\\infty)',
      'Union(Interval.open(-oo,-4),Interval.open(6,oo))',
      lambda: domain_of((x + 2) / sp.sqrt(x ** 2 - 2 * x - 24))),
    # 27 algebra/test/919
    v('algebra/test/919', 'wording', 'Let $f(x)=\\dfrac{a}{x+2}$. Find the value of $a$ for which $f(0)$ equals '
      '$f^{-1}(3a)$.', '-\\frac{10}{3}', '-10/3', lambda: solve_a(2, 3, 0)),
    v('algebra/test/919', 'numbers', 'If $f(x)=\\dfrac{a}{x+3}$, solve for the value of $a$ so that $f(0)=f^{-1}(2a)$.',
      '-\\frac{15}{2}', '-15/2', lambda: solve_a(3, 2, 0)),
    v('algebra/test/919', 'condition', 'If $f(x)=\\dfrac{a}{x+2}$, solve for the value of $a$ so that '
      '$f(1)=f^{-1}(3a)$.', '-5', '-5', lambda: solve_a(2, 3, 1)),
    # 28 intermediate_algebra/test/480
    v('intermediate_algebra/test/480', 'wording', 'Suppose $0 < a < b < x < 2a.$  Find $x$ in terms of $a$ and $b$ if\n'
      '\\[\\frac{x}{x - a} + \\frac{x - b}{x - a - b} = \\frac{x - a}{x - 2a} + \\frac{x + a - b}{x - b}.\\]',
      '\\frac{2a + b}{2}', '(2*a + b)/2', lambda: 'SYMBOLIC_480'),
    v('intermediate_algebra/test/480', 'numbers', 'Solve for $x,$ where\n\\[\\frac{x}{x - 3} + \\frac{x - 4}{x - 7} = '
      '\\frac{x - 3}{x - 6} + \\frac{x - 1}{x - 4}.\\]Assume that $6 > x > 4.$', '5', '5',
      lambda: (lambda s: s[0] if len(s) == 1 else None)(
          [r for r in sp.solve(sp.Eq(x / (x - 3) + (x - 4) / (x - 7), (x - 3) / (x - 6) + (x - 1) / (x - 4)), x)
           if 4 < r < 6])),
    v('intermediate_algebra/test/480', 'condition', 'Solve for $x,$ where\n\\[\\frac{x + a}{x - a} + \\frac{x + b}{x - b} '
      '= 2.\\]Assume that $b > x > a > 0.$', '\\frac{2ab}{a + b}', '2*a*b/(a + b)', lambda: 'SYMBOLIC_480C'),
    # 29 intermediate_algebra/test/362
    v('intermediate_algebra/test/362', 'wording', 'A sequence starts with $a_1 = 14$ and continues by the rule '
      '$a_n = 24 - 5a_{n - 1}$ for each $n \\ge 2.$  Its terms can be written as $a_n = p \\cdot q^n + r$ for some '
      'constants $p,$ $q,$ $r.$  What is $p + q + r$?', '-3', '-3', lambda: sum(recurrence_pqr(14, 24, -5))),
    v('intermediate_algebra/test/362', 'numbers', 'The sequence $(a_n)$ is defined by $a_1 = 10$ and\n\\[a_n = 12 - '
      '2a_{n - 1}\\]for all $n \\ge 2.$  Then the formula for the $n$th term can be expressed in the form '
      '$a_n = p \\cdot q^n + r,$ where $p,$ $q,$ and $r$ are constants.  Find $p + q + r.$', '-1', '-1',
      lambda: sum(recurrence_pqr(10, 12, -2))),
    v('intermediate_algebra/test/362', 'condition', 'The sequence $(a_n)$ is defined by $a_1 = 14$ and\n\\[a_n = 24 - '
      '5a_{n - 1}\\]for all $n \\ge 2.$  Then the formula for the $n$th term can be expressed in the form '
      '$a_n = p \\cdot q^n + r,$ where $p,$ $q,$ and $r$ are constants.  Find $pqr.$', '40', '40',
      lambda: sp.prod(recurrence_pqr(14, 24, -5))),
    # 30 intermediate_algebra/test/415
    v('intermediate_algebra/test/415', 'wording', 'The polynomial $x^{98} + x^{97} + \\dots + x + 1$ has roots '
      '$r_1,$ $r_2,$ $\\dots,$ $r_{98}.$  Compute the sum of $\\frac{r_k^2}{r_k + 1}$ over $k = 1,$ $2,$ $\\dots,$ $98.$',
      '-50', '-50', lambda: unity_sum(98, lambda r: r * r / (r + 1))),
    v('intermediate_algebra/test/415', 'numbers', 'Let $r_1,$ $r_2,$ $\\dots,$ $r_{50}$ be the roots of\n\\[x^{50} + '
      'x^{49} + x^{48} + \\dots + x^2 + x + 1 = 0.\\]Find\n\\[\\frac{r_1^2}{r_1 + 1} + \\frac{r_2^2}{r_2 + 1} + \\dots + '
      '\\frac{r_{50}^2}{r_{50} + 1}.\\]', '-26', '-26', lambda: unity_sum(50, lambda r: r * r / (r + 1))),
    v('intermediate_algebra/test/415', 'condition', 'Let $r_1,$ $r_2,$ $\\dots,$ $r_{98}$ be the roots of\n\\[x^{98} + '
      'x^{97} + x^{96} + \\dots + x^2 + x + 1 = 0.\\]Find\n\\[\\frac{1}{r_1 + 1} + \\frac{1}{r_2 + 1} + \\dots + '
      '\\frac{1}{r_{98} + 1}.\\]', '49', '49', lambda: unity_sum(98, lambda r: 1 / (r + 1))),
]


def symbolic_checks():
    """Symbolic answers of family intermediate_algebra/test/480: verified at many random (a, b) in the stated range."""
    import random
    a, b = sp.symbols('a b', positive=True)
    rnd = random.Random(1)
    lhs = lambda X, A, B: X / (X - A) + (X - B) / (X - A - B) - (X - A) / (X - 2 * A) - (X + A - B) / (X - B)
    lhs_c = lambda X, A, B: (X + A) / (X - A) + (X + B) / (X - B) - 2
    for _ in range(25):
        A = sp.Rational(rnd.randint(10, 40), 7)
        B = A * sp.Rational(rnd.randint(11, 19), 10)  # a < b < 2a
        sols = [s for s in sp.solve(sp.Eq(lhs(x, A, B), 0), x) if s.is_real and B < s < 2 * A]
        assert sols == [(2 * A + B) / 2], (A, B, sols)
        B2 = A * sp.Rational(rnd.randint(11, 40), 10)  # a < b
        sols = [s for s in sp.solve(sp.Eq(lhs_c(x, A, B2), 0), x) if s.is_real and A < s < B2]
        assert sols == [2 * A * B2 / (A + B2)], (A, B2, sols)
    return {'SYMBOLIC_480': (2 * a + b) / 2, 'SYMBOLIC_480C': 2 * a * b / (a + b)}
