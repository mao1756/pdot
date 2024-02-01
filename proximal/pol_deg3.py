import math


class MonComplexe:
    """modélisation des nombres complexes"""

    def __init__(self, a=0.0, b=0.0):
        """le constructeur soit en cartésiennes soit en polaires"""
        self.x = a
        self.y = b

    def __str__(self):
        """représentation externe pour print et str"""
        if self.y == 0:
            return str(self.x)
        if self.x == 0 and self.y == 1:
            return "i"
        if self.x == 0 and self.y == -1:
            return "-i"
        if self.x == 0:
            return str(self.y) + "i"
        if self.y == 1:
            return str(self.x) + "+i"
        if self.y == -1:
            return str(self.x) + "-i"
        if self.y > 0:
            return str(self.x) + "+" + str(self.y) + "i"
        else:
            return str(self.x) + str(self.y) + "i"

    def __add__(self, other):
        """somme de deux complexes"""
        a = self.x + other.x
        b = self.y + other.y
        return MonComplexe(a, b)

    def __sub__(self, other):
        """différence de deux complexes"""
        a = self.x - other.x
        b = self.y - other.y
        return MonComplexe(a, b)

    def __neg__(self):
        """opposé d'un complexe"""
        return MonComplexe(-self.x, -self.y)

    def null(self):
        """test de nullité"""
        return self.x == 0 and self.y == 0

    def __mul__(self, other):
        """produit de deux complexes"""
        a = self.x * other.x - self.y * other.y
        b = self.x * other.y + self.y * other.x
        return MonComplexe(a, b)

    def __truediv__(self, other):
        """quotient de deux complexes"""
        return self * (~other)

    def conj(self):
        """conjugué d'un complexe"""
        return MonComplexe(self.x, -self.y)

    def module(self):
        """module d'un complexe"""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def argument(self):
        """argument d'un complexe"""
        if self.x == 0 and self.y == 0:
            return 0
        if self.x == 0 and self.y > 0:
            return math.pi / 2
        if self.x == 0 and self.y < 0:
            return -math.pi / 2
        if self.x > 0:
            return math.atan(self.y / self.x)
        return math.pi - math.atan(self.y / (-self.x))

    def __invert__(self):
        """inverse d'un complexe"""
        if self.null():
            raise ZeroDivisionError
        return MonComplexe(
            self.x / (self.x * self.x + self.y * self.y),
            -self.y / (self.x * self.x + self.y * self.y),
        )

    def __pow__(self, n):
        """puissances d'un complexe"""
        if n <= 0 and self.null():
            raise ZeroDivisionError
        if n >= 0:
            R = MonComplexe(1, 0)
            while n:
                R = R * self
                n = n - 1
            return R
        if n < 0:
            return (~R) ** (-n)

    def racines(self, n):
        """calcule les n racines n-ièmes du nombre"""
        # on utilise les racines de l'unité
        return [
            MonComplexe(
                self.module() ** (1.0 / n)
                * math.cos((k * 2 * math.pi + self.argument()) / n),
                self.module() ** (1.0 / n)
                * math.sin((k * 2 * math.pi + self.argument()) / n),
            )
            for k in range(0, n)
        ]


def Cardan(a, b, c, d):
    """a, b, c, d sont les coefficients initiaux de l'équation"""
    # on commence par mettre sous forme canonique
    b, c, d = b / a, c / a, d / a
    p = c - b * b / MonComplexe(3.0, 0)
    q = (
        d
        - b * c / MonComplexe(3.0, 0)
        - (b**3) / MonComplexe(27.0, 0)
        + (b**3) / MonComplexe(9.0, 0)
    )
    B, C = q, -p * p * p / MonComplexe(27.0, 0)
    D = B * B - MonComplexe(4.0, 0) * C
    R = D.racines(2)
    U = (-B + R[0]) / MonComplexe(2.0, 0)
    roots = U.racines(3)
    sol1 = [u - p / (MonComplexe(3.0, 0) * u) for u in roots]
    sol2 = [z - b / MonComplexe(3.0, 0) for z in sol1]
    return sol2


def Resolution(a, b, c, d):
    """Résout l'équation az^3+bz^2+c^z+d=0"""
    # les coefficients peuvent être entiers, réels ou complexes
    # Dans tous les cas on convertit en complexes pour commencer
    if isinstance(a, float) or isinstance(a, int):
        a = MonComplexe(float(a), 0)
    if isinstance(b, float) or isinstance(b, int):
        b = MonComplexe(float(b), 0)
    if isinstance(c, float) or isinstance(c, int):
        c = MonComplexe(float(c), 0)
    if isinstance(d, float) or isinstance(d, int):
        d = MonComplexe(float(d), 0)
    Z = Cardan(a, b, c, d)
    return Z


def maxSol(a, b, c, d):
    z = Resolution(a, b, c, d)
    l = []
    if abs(z[0].y) < 1e-7:
        l.append(z[0].x)
    if abs(z[1].y) < 1e-7:
        l.append(z[1].x)
    if abs(z[2].y) < 1e-7:
        l.append(z[2].x)
    return max(l)


def main():
    a, b, c, d = 1, 3, -6, 4
    Resolution(a, b, c, d)
    a, b, c, d = 1, 0, MonComplexe(0, -6.0), MonComplexe(4.0, -4.0)
    Resolution(a, b, c, d)
    a, b, c, d = 1, 0, -4, 2
    Resolution(a, b, c, d)


if __name__ == "__main__":
    main()
