"""Symbolic proofs for the 628 and 15 families (stage 2 variants).
628: AM-HM pair bound 1/(x+y) <= (1/x+1/y)/4, summed over pairs, equality at all-equal.
15:  rotate (x, z) -> (u, v); R^2*sqrt(k+1)/2 - y(sqrt(k)x+z) = sqrt(k+1)/2*((y-u)^2+v^2)."""
import itertools, json, sympy as sp
out = {}
x, y = sp.symbols('x y', positive=True)
gap = sp.factor((1/x + 1/y)/4 - 1/(x + y))
assert sp.simplify(gap - (x - y)**2/(4*x*y*(x + y))) == 0
out["628_pair_bound"] = f"(1/x+1/y)/4 - 1/(x+y) = {gap} >= 0"
for n, claim in ((3, sp.Integer(2)), (4, sp.Rational(4, 3))):
    v = sp.symbols(f'a1:{n+1}', positive=True)
    num = sum(1/t for t in v)
    den = sum(1/(p + q) for p, q in itertools.combinations(v, 2))
    # each variable lies in n-1 pairs, so den <= (n-1)/4 * num, ratio >= 4/(n-1)
    bound = sp.Rational(4, n - 1)
    slack = sp.simplify(sp.Rational(n - 1, 4)*num - den
                        - sum((p - q)**2/(4*p*q*(p + q)) for p, q in itertools.combinations(v, 2)))
    assert slack == 0 and bound == claim
    at_eq = sp.simplify((num/den).subs({t: 1 for t in v}))
    assert at_eq == claim
    out[f"628_n{n}"] = f"num/den >= {bound}, attained at all equal (value {at_eq}); reciprocal max = {1/bound}"
k, R = sp.symbols('k R', positive=True)
X, Y, Z = sp.symbols('X Y Z', real=True)
u = (sp.sqrt(k)*X + Z)/sp.sqrt(k + 1)
w = (X - sp.sqrt(k)*Z)/sp.sqrt(k + 1)
lhs = sp.sqrt(k + 1)/2*(X**2 + Y**2 + Z**2) - Y*(sp.sqrt(k)*X + Z)
rhs = sp.sqrt(k + 1)/2*((Y - u)**2 + w**2)
assert sp.simplify(sp.expand(lhs - rhs)) == 0
out["15_identity"] = "sqrt(k+1)/2*(x^2+y^2+z^2) - y(sqrt(k)x+z) = sqrt(k+1)/2*((y-u)^2+v^2)"
for kk, RR, claim in ((10, 1, sp.sqrt(11)/2), (8, 1, sp.Rational(3, 2)), (10, 2, 2*sp.sqrt(11))):
    # equality: v=0 -> x:z = sqrt(k):1, y = u, x^2+y^2+z^2 = R^2
    yy = RR/sp.sqrt(2); xx = sp.sqrt(kk)*yy/sp.sqrt(kk + 1); zz = yy/sp.sqrt(kk + 1)
    assert sp.simplify(xx**2 + yy**2 + zz**2 - RR**2) == 0 and min(xx, yy, zz) > 0
    val = sp.simplify(yy*(sp.sqrt(kk)*xx + zz))
    assert sp.simplify(val - claim) == 0 and sp.simplify(sp.sqrt(kk + 1)*RR**2/2 - claim) == 0
    out[f"15_k{kk}_R{RR}"] = f"max = sqrt({kk+1})*{RR}^2/2 = {claim}, attained at positive (x,y,z)=({xx}, {yy}, {zz})"
print(json.dumps(out, indent=1, ensure_ascii=False))
