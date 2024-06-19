## Spherical Harmonics

This section provides documentation for the evaluation of complex and real spherical harmonics and solid harmonics, including gradient and Laplacian calculations. 

- [Associated Legendre Polynomials](#Associated-Legendre-Polynomials)
- [Complex Spherical Harmonics](#Complex-Spherical-Harmonics)
- [Complex Solid Harmonics](#Complex-Solid-Harmonics)
- [Real Spherical Harmonics](#Real-Spherical-Harmonics)
- [Real Solid Harmonics](#Real-Solid-Harmonics)

### Associated Legendre Polynomials
Associated Legendre polynomials ``P_{\ell}^m`` are closely related to the spherical harmonics, ``P_{\ell}^m`` of degree ``\ell`` and order ``m\geq 0`` are defined as (in the phase convention of Condon and Shortley)
```math 
	P_{\ell}^m(x) = \frac{(-1)^m}{2^{\ell}\ell!}(1-x^2)^{m/2}\frac{\mathrm{d}^{\ell+m}}{\mathrm{d}x^{\ell+m}}(x^2-1)^{\ell}.
```
The negative order can be related to the corresponding positive order via a proportionality constant that involves only ``\ell`` and ``m``, 
```math 
P_{\ell}^{-m}(x) = (-1)^m \frac{(\ell-m)!}{(\ell+m)!}P_{\ell}^m(x). 
```
The associated Legendre polynomials are orthogonal on the interval ``-1\leq x\leq 1`` in the sense that 
```math 
\int_{-1}^{1} P_{k}^m(x) P_{\ell}^m(x) \mathrm{d}x = \frac{2}{2\ell+1}\frac{(\ell+m)!}{(\ell-m)!}\delta_{k\ell}. 
```

In `alp.jl`, Polynomials4ML utilizes the following normalization for the associated Legendre polynomials,  
```math 
\bar{P}_{\ell}^{m}(x) = \sqrt{\frac{(2\ell+1)(\ell-m)!}{2\pi (\ell+m)!}}P_{\ell}^m, \qquad m\geq 0
```
and one can generate a data structure as 
```julia
ALPs = ALPolynomials(maxL::Integer, T::Type=Float64)
```
where `maxL` specifies the maximum degree of the polynomials. 

The associated Legendre polynomials allow for 
```julia
P = evaluate(basis, X)
P, dP = evaluate_ed(basis, X)
```
`X` is a point in spherical coordinates, `P` and `dp` stores `P_l^m(X.cosθ)` and `dP_l^m(X.cosθ)`. Specifically, only non-negative `m` terms are stored, and are arranged in $\ell$-major order. To retrieve the specific values of `P_l^m` and `dP_l^m` for given indices `(l, m)`, one can use
```julia
index_p(l,m)
```
The algorithm for computing associated Legendre polynomials is based on Dusson(2022) eq.(A.7), where `A_l^m`, `B_l^m`, `C_l^m` can be found in Limpanuparb(2014) eq.(7)-(14). 

### Condon-Shortley Sign Convention
There are two sign conventions for associated Legendre polynomials. 
- Include the Condon-Shortley phase factor:
```math
P_{\ell}^m(x) = \frac{(-1)^m}{2^{\ell}\ell!}(1-x^2)^{m/2}\frac{\mathrm{d}^{\ell+m}}{\mathrm{d}x^{\ell+m}}(x^2-1)^{\ell}.
```
- Exclude the Condon-Shortley phase factor:
```math
P_{\ell}^m(x) = \frac{1}{2^{\ell}\ell!}(1-x^2)^{m/2}\frac{\mathrm{d}^{\ell+m}}{\mathrm{d}x^{\ell+m}}(x^2-1)^{\ell}.
```
One possible way to distinguish the two conventions is
```math
P_{\ell m}(x) = (-1)^m P_{\ell}^m(x). 
```

The Condon-Shortley sign convention enables us to establish the following relationships between spherical harmonics and angular momentum ladder operators
```math
Y_{\ell}^m(\theta, \varphi) = A_{\ell m}\hat{L}_-^{\ell-m}Y_{\ell}^{\ell}(\theta, \varphi), 
```

```math
Y_{\ell}^m(\theta, \varphi) = A_{\ell, -m}\hat{L}_+^{\ell+m}Y_{\ell}^{-\ell}(\theta, \varphi), 
```
with all positive constants ``A_{\ell m} = \sqrt{\frac{(\ell+m)!}{(2\ell)!(\ell+m)!}}``. Ignoring the Condon-Shortley phase would introduce signs into the ``A_{\ell m}``. It's only a sign convention. 

Including the factor of ``(-1)^m`` and written in terms ``x=\cos\theta``,  the first few associated Legendre polynomials are 

| ``m\backslash\ell``| 0         | 1             | 2                                 | 3                                             |
|--------------------|-----------|---------------|-----------------------------------|-----------------------------------------------|
| 3                  |           |               |                                   | ``-15\sin^3\theta``                           |
| 2                  |           |               | ``3\sin^2\theta``                 |``15\cos\theta\sin^2\theta``                   |
| 1                  |           |``-\sin\theta``| ``-3\sin\theta\cos\theta``        | ``-\frac{3}{2}(5\cos^2\theta - 1)\sin\theta`` |
| 0                  | ``1``     | ``\cos\theta``| ``\frac{1}{2}(3\cos^2\theta - 1)``| ``\frac{1}{2}\cos\theta(5\cos^2\theta-3)``    |


### Complex Spherical Harmonics
In `cylm.jl`, Polynomials4ML utilizes orthonormalized complex spherical harmonics that includes the Condon-Shortley phase, defined as
```math
	Y_{\ell}^m(\theta, \varphi) = \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}}P_{\ell}^m(\cos \theta)\mathrm{e}^{\mathrm{i}m \varphi}. 
```
The normalization in complex spherical harmonics is chosen to satisfy
```math
	\int_0^{2\pi}\int_0^{\pi}Y_{k}^m(\theta, \varphi)\bar{Y}_{\ell}^n(\theta, \varphi)\sin \theta \mathrm{d}\theta\mathrm{d}\varphi =\delta_{k\ell}\delta_{mn}.
```
Orthonormalized complex spherical harmonics that include the Condon-Shortley phase up to degree ``\ell = 3`` are

| $m\backslash\ell$ | 0         | 1          | 2         | 3          |
|------------------|-----------|-----------|-----------|-----------|
| 3                |           |           |           | $-\frac{1}{8}\sqrt{\frac{35}{\pi}}\cdot \mathrm{e}^{3\mathrm{i}\varphi}\cdot \sin^3\theta$ |
| 2                |           |           | $\frac{1}{4}\sqrt{\frac{15}{2\pi}}\cdot \mathrm{e}^{2\mathrm{i}\varphi}\cdot \sin^2\theta$ | $\frac{1}{4}\sqrt{\frac{105}{2\pi}}\cdot \mathrm{e}^{2\mathrm{i}\varphi}\cdot \sin^2\theta\cdot \cos\theta$ |
| 1                |           | $-\frac{1}{2}\sqrt{\frac{3}{2\pi}}\cdot \mathrm{e}^{\mathrm{i}\varphi}\cdot \sin\theta$ | $-\frac{1}{2}\sqrt{\frac{15}{2\pi}}\cdot \mathrm{e}^{\mathrm{i}\varphi}\cdot \sin\theta \cdot \cos\theta$ | $-\frac{1}{8}\sqrt{\frac{21}{\pi}}\cdot \mathrm{e}^{\mathrm{i}\varphi}\cdot \sin\theta \cdot (5\cos^2\theta-1)$ |
| 0                | $\frac{1}{2}\sqrt{\frac{1}{\pi}}$ | $\frac{1}{2}\sqrt{\frac{3}{\pi}}\cdot \cos \theta$ | $\frac{1}{4}\sqrt{\frac{5}{\pi}}\cdot (3\cos^2\theta - 1)$ | $\frac{1}{4}\sqrt{\frac{7}{\pi}}\cdot (5\cos^3\theta - 3\cos\theta)$ |
| -1               |           | $\frac{1}{2}\sqrt{\frac{3}{2\pi}}\cdot \mathrm{e}^{-\mathrm{i}\varphi}\cdot \sin\theta$ | $\frac{1}{2}\sqrt{\frac{15}{2\pi}}\cdot \mathrm{e}^{-\mathrm{i}\varphi}\cdot \sin\theta \cdot \cos\theta$ | $\frac{1}{8}\sqrt{\frac{21}{\pi}}\cdot \mathrm{e}^{-\mathrm{i}\varphi}\cdot \sin\theta \cdot (5\cos^2\theta-1)$ |
| -2               |           |           | $\frac{1}{4}\sqrt{\frac{15}{2\pi}}\cdot \mathrm{e}^{-2\mathrm{i}\varphi}\cdot \sin^2\theta$ | $\frac{1}{4}\sqrt{\frac{105}{2\pi}}\cdot \mathrm{e}^{-2\mathrm{i}\varphi}\cdot \sin^2\theta\cdot \cos\theta$ |
| -3               |           |           |           | $\frac{1}{8}\sqrt{\frac{35}{\pi}}\cdot \mathrm{e}^{-3\mathrm{i}\varphi}\cdot \sin^3\theta$|

To generate the complex spherical harmonics ``Y_{\ell}^m`` with normalized associated Legendre polynomials ``\bar{P}_{\ell}^m``, the formulas can be rewritten as
```math
\begin{cases}
Y_{\ell}^0(\theta, \varphi) = \sqrt{\frac{1}{2}}\bar{P}_{\ell}^0(\cos \theta)\\
Y_{\ell}^m(\theta, \varphi) = \sqrt{\frac{1}{2}}\bar{P}_{\ell}^m(\cos \theta)\mathrm{e}^{\mathrm{i}m \varphi}\\
Y_{\ell}^{-m}(\theta, \varphi) = (-1)^m\cdot \sqrt{\frac{1}{2}} \bar{P}_{\ell}^m(\cos \theta)\mathrm{e}^{-\mathrm{i}m \varphi}
\end{cases}, \qquad m>0.
```
To evaluate the gradients of the spherical harmonics ``\nabla Y_{\ell}^m``, one need to convert a gradient with respect to spherical coordinates to a gradient with respect to cartesian coordinates, 
```math
\begin{cases}
(\frac{\partial \varphi}{\partial x}, \frac{\partial \varphi}{\partial y}, \frac{\partial \varphi}{\partial z}) = (-\frac{\sin\varphi}{r\sin \theta}, \frac{\cos\varphi}{r\sin \theta}, 0)\\
(\frac{\partial \theta}{\partial x}, \frac{\partial \theta}{\partial y}, \frac{\partial \theta}{\partial z}) = (\frac{\cos \varphi \cos \theta}{r}, \frac{\sin \varphi \cos \theta}{r}, -\frac{\sin \theta}{r})
\end{cases}.
```
Therefore, the gradient of ``Y_{\ell}^m`` can be expressed as,
```math
\nabla Y_{\ell}^m = \frac{\mathrm{i}m P_{\ell}^m \mathrm{e}^{\mathrm{i}m\varphi}}{r\sin \theta}\begin{bmatrix} -\sin \varphi \\\cos \varphi \\0 \end{bmatrix} + 
\frac{\partial_{\theta}P_{\ell}^m \mathrm{e}^{\mathrm{i}m\varphi}}{r}\begin{bmatrix} \cos \varphi \cos \theta\\\sin \varphi\cos \theta \\-\sin\theta \end{bmatrix}.
```
For the sake of simplicity, we incorporated the coefficient in of ``P_{\ell}^m`` into the term ``P_{\ell}^m`` itself.

To ensure numerically stable evaluation of gradients near ``\sin \theta = 0``, we compute ``P_{\ell}^m/\sin \theta`` instead of ``P_{\ell}^m``. We refer to section A.1 of Dusson(2022) for detailed discussion.

We can further compute ``\nabla^2 Y_{\ell}^m`` as, 
```math
\nabla^2 Y_{\ell m} = \left(\frac{1}{r^2}\frac{\partial}{\partial r} r^2\frac{\partial}{\partial r} - \frac{L^2}{r^2}    \right)Y_{\ell}^m = -\frac{\ell(\ell+1)}{r^2}Y_{\ell}^{m}.
```
One can generate a data structure as 
```julia
cylm = CYlmBasis(maxL::Integer, T::Type=Float64)
```
The complex spherical harmonics allow for 
```julia
P = evaluate(basis, X)
P, dP = evaluate_ed(basis, X)
```
To retrieve the specific values of `Y_l^m` and `dY_l^m` for given indices `(l, m)`, one can use
```julia
index_y(l,m)
```
### Alternative normalizations conventions
Here, we provide a list of alternative normalizations conventions for complex spherical harmonics,

-  Schmidt semi-normalized (Racah's normalization)
```math
C_{\ell}^m(\theta, \varphi) = \sqrt{\frac{4\pi}{2\ell + 1}}Y_{\ell}^m(\theta, \varphi) = \sqrt{\frac{(\ell-m)!}{(\ell+m)!}}P_{\ell}^m(\cos \theta)\mathrm{e}^{\mathrm{i}m \varphi}, 
```
with 
```math
\int_0^{2\pi}\int_0^{\pi}C_{k}^m(\theta, \varphi)\bar{C}_{\ell}^n(\theta, \varphi)\sin \theta \mathrm{d}\theta\mathrm{d}\varphi = \frac{4\pi}{2\ell + 1}\delta_{k\ell}\delta_{mn}. 
```
In this normalization, ``C_0^0(\theta, \varphi)`` is equal to $1$. 

- 4π-normalized
```math
\mathscr{Y}_{\ell}^m (\theta, \varphi) = \sqrt{4\pi}Y_{\ell}^m(\theta, \varphi) = \sqrt{(2\ell+1)\frac{(l-m)!}{(l+m)!}}P_{\ell}^m(\cos \theta)\mathrm{e}^{\mathrm{i}m \varphi}, 
```
with 
```math
\int_0^{2\pi}\int_0^{\pi}\mathscr{Y}_{k}^m(\theta, \varphi)\bar{\mathscr{Y}}_{\ell}^n(\theta, \varphi)\sin \theta \mathrm{d}\theta\mathrm{d}\varphi = 4\pi\delta_{k\ell}\delta_{mn}. 
```
### Complex Solid Harmonics
In `crlm.jl`, Polynomials4ML utilizes orthonormalized complex solid harmonics defined as
```math
	\gamma_{\ell}^m(r, \theta, \varphi) = r^{\ell}Y_{\ell}^m(\theta, \varphi). 
```
``\gamma_{\ell}^m``'s are orthogonal is the sense that 
```math
	\int_0^{2\pi}\int_0^{\pi}\gamma_{k}^m(\theta, \varphi)\bar{\gamma}_{\ell}^n(\theta, \varphi)\sin \theta \mathrm{d}\theta\mathrm{d}\varphi =\delta_{k\ell}\delta_{mn}r^{k+\ell}.
```
The evaluation of solid harmonics can be obtained from the spherical harmonics by a simple scaling with ``r^{\ell}``. 
To evaluate the gradients of the solid harmonics, ``\nabla \gamma_{\ell}^m``, the following expressions are used,
```math
\begin{cases}
(\frac{\partial r}{\partial x}, \frac{\partial r}{\partial y}, \frac{\partial r}{\partial z}) = (\sin \theta \cos \varphi,\sin\theta\sin \varphi, \cos \theta)\\
(\frac{\partial \varphi}{\partial x}, \frac{\partial \varphi}{\partial y}, \frac{\partial \varphi}{\partial z}) = (-\frac{\sin\varphi}{r\sin \theta}, \frac{\cos\varphi}{r\sin \theta}, 0)\\
(\frac{\partial \theta}{\partial x}, \frac{\partial \theta}{\partial y}, \frac{\partial \theta}{\partial z}) = (\frac{\cos \varphi \cos \theta}{r}, \frac{\sin \varphi \cos \theta}{r}, -\frac{\sin \theta}{r})
\end{cases}.
```
Therefore, the gradient of ``\gamma_{\ell}^m`` can be expressed as,
```math
\nabla \gamma_{\ell}^m = \frac{\ell r^{\ell} P_{\ell}^m \mathrm{e}^{\mathrm{i}m\varphi}}{r}\begin{bmatrix} \sin \theta \cos \varphi\\ \sin\theta\sin \varphi\\ \cos \theta \end{bmatrix}+ \frac{\mathrm{i}m P_{\ell}^m \mathrm{e}^{\mathrm{i}m\varphi}}{r\sin \theta}\begin{bmatrix} -\sin \varphi \\\cos \varphi \\0 \end{bmatrix} + 
\frac{\partial_{\theta}P_{\ell}^m \mathrm{e}^{\mathrm{i}m\varphi}}{r}\begin{bmatrix} \cos \varphi \cos \theta\\\sin \varphi\cos \theta \\-\sin\theta \end{bmatrix}.
```
Similarly, we incorporated the coefficient in of ``P_{\ell}^m`` into the term ``P_{\ell}^m`` itself.
We can further compute ``\nabla^2 \gamma_{\ell}^m`` as, 
```math
\nabla^2 r^{\ell}Y_{\ell m} = \left(\frac{1}{r^2}\frac{\partial}{\partial r} r^2\frac{\partial}{\partial r} - \frac{L^2}{r^2}    \right)r^{\ell}Y_{\ell}^m = \frac{Y_{\ell m}}{r^2}\frac{\partial}{\partial r}r^2\frac{\partial r^{\ell}}{\partial r} - \frac{r^{\ell}L^2 Y_{\ell}^{m}}{r^2} = 0,
```
that is, the solid harmonics are solutions to Laplace's equation. 
### Real Spherical Harmonics
In `rylm.jl`, Polynomials4ML utilizes orthonormalized real spherical harmonics that exclude the Condon-Shortley phase. 
		 
- Include the Condon-Shortley phase factor:
```math
Y_{\ell m}(\theta, \varphi)  = 
\begin{cases}
\frac{\mathrm{i}}{\sqrt{2}}(Y_{\ell}^m - (-1)^m Y_{\ell}^{-m}) & m < 0\\
Y_{\ell}^0 & m = 0 \\
\frac{1}{\sqrt{2}}(Y_{\ell}^{-m} + (-1)^m Y_{\ell}^{m}) & m > 0
\end{cases} = 
\begin{cases}
(-1)^m \bar{P}_{\ell}^{|m|}(\cos \theta)\sin(|m|\varphi) & m < 0\\
\frac{1}{\sqrt{2}} \bar{P}_{\ell}^0(\cos \theta) & m = 0 \\
(-1)^m \bar{P}_{\ell}^{m}(\cos \theta)\cos(m\varphi)  & m > 0
\end{cases}
```

- Exclude the Condon-Shortley phase factor:
```math
Y_{\ell m}(\theta, \varphi) = 
\begin{cases}
-\bar{P}_{\ell}^{|m|}(\cos \theta)\sin(|m|\varphi) & m < 0\\
\frac{1}{\sqrt{2}} \bar{P}_{\ell}^0(\cos \theta) & m = 0 \\
\bar{P}_{\ell}^{m}(\cos \theta)\cos(m\varphi)  & m > 0
\end{cases}
```

Orthonormalized real spherical harmonics that employ the Condon-Shortley phase up to degree ``\ell = 3`` are

| ``m\backslash\ell``| 0                                 | 1         | 2         | 3         |
|--------------------|-----------------------------------|-----------|-----------|-----------|
| 3                  |                                   |           |           | ``\frac{1}{4}\sqrt{\frac{35}{2\pi}}\cdot \frac{x(x^2-3y^2)}{r^3}`` |
| 2                  |                                   |           | ``\frac{1}{4}\sqrt{\frac{15}{\pi}}\cdot \frac{x^2-y^2}{r^2}`` | ``\frac{1}{4}\sqrt{\frac{105}{\pi}}\cdot \frac{(x^2-y^2)z}{r^3}`` |
| 1                  |                                   | ``\sqrt{\frac{3}{4\pi}}\cdot \frac{x}{r}`` | ``\frac{1}{2}\sqrt{\frac{15}{\pi}}\cdot \frac{zx}{r^2}`` | ``\frac{1}{4}\sqrt{\frac{21}{2\pi}}\cdot \frac{x(5z^2-r^2)}{r^3}`` |
| 0                  |``\frac{1}{2}\sqrt{\frac{1}{\pi}}``| ``\sqrt{\frac{3}{4\pi}}\cdot \frac{z}{r}`` | ``\frac{1}{4}\sqrt{\frac{5}{\pi}}\cdot \frac{3z^2-r^2}{r^2}`` | ``\frac{1}{4}\sqrt{\frac{7}{\pi}}\cdot \frac{z(5z^2-3r^2)}{r^3}`` |
| -1                 |                                   | ``\sqrt{\frac{3}{4\pi}}\cdot \frac{y}{r}``  | ``\frac{1}{2}\sqrt{\frac{15}{\pi}}\cdot \frac{yz}{r^2}`` | ``\frac{1}{4}\sqrt{\frac{21}{2\pi}}\cdot \frac{y(5z^2-r^2)}{r^3}`` |
| -2                 |                                   |           | ``\frac{1}{2}\sqrt{\frac{15}{\pi}}\cdot \frac{xy}{r^2}`` |``\frac{1}{2}\sqrt{\frac{105}{\pi}}\cdot \frac{xyz}{r^2}`` |
| -3                 |                                   |           |           | ``\frac{1}{4}\sqrt{\frac{35}{2\pi}}\cdot \frac{(3x^2-y^2)y}{r^3}``|

### Real Solid Harmonics
In `rrlm.jl`, Polynomials4ML utilizes Schmidt semi-normalized real solid harmonics that exclude the Condon-Shortley phase.

- Include the Condon-Shortley phase factor:
```math
S_{\ell m}(r, \theta, \varphi)  = 
\begin{cases}
\frac{\mathbb{i}}{\sqrt{2}}\left(C_{\ell, m}-(-1)^m C_{\ell,-m} \right) & m < 0\\
C_{10} & m = 0 \\
 \frac{1}{\sqrt{2}}\left(C_{\ell, -m}+(-1)^m C_{\ell,m}\right)  & m > 0
\end{cases} = \begin{cases}
(-1)^m \sqrt{\frac{4\pi}{2l+1}}\cdot r^{\ell}\bar{P}_{\ell}^{|m|}(\cos \theta)\sin(|m|\varphi) & m < 0\\
\sqrt{\frac{2\pi}{2l+1}}\bar{P}_{\ell}^0(\cos \theta) & m = 0 \\
(-1)^m \sqrt{\frac{4\pi}{2l+1}}\cdot r^{\ell}\bar{P}_{\ell}^{m}(\cos \theta)\cos(m\varphi)  & m > 0
\end{cases},
```
where 
```math
C_{\ell, m}(r, \theta, \varphi) = \sqrt{\frac{4\pi}{2\ell + 1}}\gamma_{\ell}^m(\theta, \varphi), 
```
with 
```math
\int_0^{2\pi}\int_0^{\pi}C_{k,m}(r, \theta, \varphi)\bar{C}_{\ell, n}(r, \theta, \varphi)\sin \theta \mathrm{d}\theta\mathrm{d}\varphi = \frac{4\pi}{2\ell + 1}\delta_{k\ell}\delta_{mn} r^{k+\ell}. 
```

- Exclude the Condon-Shortley phase factor:
```math
S_{\ell m}(r, \theta, \varphi)  = 
\begin{cases}
-\sqrt{\frac{4\pi}{2l+1}}\cdot r^{\ell}\bar{P}_{\ell}^{|m|}(\cos \theta)\sin(|m|\varphi) & m < 0\\
\sqrt{\frac{2\pi}{2l+1}}\bar{P}_{\ell}^0(\cos \theta) & m = 0 \\
\sqrt{\frac{4\pi}{2l+1}}\cdot r^{\ell}\bar{P}_{\ell}^{m}(\cos \theta)\cos(m\varphi)  & m > 0
\end{cases}
```

Schmidt semi-normalized real spherical harmonics that employ the Condon-Shortley phase up to degree ``\ell = 3`` are

| ``m\backslash\ell``| 0         | 1         | 2                                | 3                                            |
|--------------------|-----------|-----------|----------------------------------|----------------------------------------------|
| 3                  |           |           |                                  | ``\frac{1}{2}\sqrt{\frac{5}{2}}(x^2-3y^2)x`` |
| 2                  |           |           | ``\frac{1}{2}\sqrt{3}(x^2-y^2)`` | ``\frac{1}{2}\sqrt{15}(x^2-y^2)z``           |
| 1                  |           | ``x``     | ``\sqrt{3}xz``                   | ``\frac{1}{2}\sqrt{\frac{3}{2}}(5z^2-r^2)x`` |
| 0                  | ``1``     | ``z``     | ``\frac{1}{2}(3z^2-r^2)``        | ``\frac{1}{2}(5z^2-3r^2)z``                  |
| -1                 |           | ``y``     | ``\sqrt{3}yz``                   | ``\frac{1}{2}\sqrt{\frac{3}{2}}(5z^2-r^2)y`` |
| -2                 |           |           | ``\sqrt{3}xy``                   |``\sqrt{15}xyz``                              |
| -3                 |           |           |                                  | ``\frac{1}{2}\sqrt{\frac{5}{2}}(3x^2-y^2)y`` |


### References

1. Dusson, G., Bachmayr, M., Csányi, G., Drautz, R., Etter, S., van der Oord, C., & Ortner, C. (2022). [Atomic cluster expansion: Completeness, efficiency and stability](https://arxiv.org/pdf/1911.03550.pdf). Journal of Computational Physics, 454, 110946.
2. Helgaker, T., Jorgensen, P., & Olsen, J. (2013). [Molecular electronic-structure theory](https://www.wiley.com/en-us/Molecular+Electronic-Structure+Theory-p-9780471967552). John Wiley & Sons.
3. Limpanuparb, T., & Milthorpe, J. (2014). [Associated Legendre polynomials and spherical harmonics computation for chemistry applications](https://arxiv.org/pdf/1410.1748.pdf). arXiv preprint arXiv:1410.1748.  
4. Wieczorek, M. A., & Meschede, M. (2018). [SHTools: Tools for working with spherical harmonics.](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018GC007529) Geochemistry, Geophysics, Geosystems, 19(8), 2574-2592.