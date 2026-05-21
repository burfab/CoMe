# GOF Backward Pass — Gradient Cheat Sheet

## Notation

| Symbol | Meaning |
|--------|---------|
| `w` | `conic_opacity[id].w` — gaussian opacity parameter |
| `G` | gaussian value at peak (evaluated at `t = -BB/(2*AA)`) |
| `alpha` | `min(0.99, w * G)` |
| `weight` | `alpha * T` |
| `test_T` | `T * (1 - alpha)` |
| `AA, BB, CC` | quadratic ray-gaussian intersection coefficients |
| `t` | gaussian peak depth: `t = -BB / (2*AA)` |
| `t_star` | exact surface depth (EXACT_DEPTH only) |
| `depth` | `max_depth` stored from forward pass |
| `p` | `min(0, -0.5 * min_value)` — clamped exponent |

---

## Prep (read from forward pass outputs)

```
T_final        = final_Ts[pix]
final_D        = final_Ts[pix + H*W]          (dist1 accumulator)
final_D2       = final_Ts[pix + 2*H*W]        (dist2 accumulator)
final_A        = 1 - T_final
opacity_final  = pixel_colors[ALPHA_OFFSET]
T_opa_final    = final_Ts[pix + 3*H*W]
max_depth      = pixel_colors[DEPTH_OFFSET]
max_contributor = n_contrib & 0xFFFF
blend_contributor = (n_contrib >> 16) & 0xFFFF
```

---

## Per-Gaussian Gradients (blend_function, front-to-back)

### 1. Color

```
dL_dcolor[ch]  += weight * dL_dpixel[ch]

accum_rec[ch]   = (final_color[ch] - C[ch]) / test_T      (reconstructed future color)
dL_dalpha      += Σ_ch (c[ch] - accum_rec[ch]) * dL_dpixel[ch]
```

### 2. Variance Loss

```
dL_dcolor[ch]  += 2 * λ_var * weight * (c[ch] - gt[ch])

D_i             = Σ_ch (c[ch] - gt[ch])²
remaining_var  -= weight * D_i
dL_dalpha      += λ_var * (T * D_i - remaining_var / (1 - alpha)) / T
```

### 3. Normal Blending

```
accum_n_rec[ch] = (final_normal[ch] - C[CHANNELS+ch]) / test_T
dL_dalpha      += Σ_ch (n_norm[ch] - accum_n_rec[ch]) * dL_dnormal[ch]
dL_dnormal_norm[ch]  = weight * dL_dnormal[ch]
```

### 4. Normal Variance Loss

```
diff[ch]        = n_norm[ch] - final_normal[ch]
dL_dnormal_norm[ch] += 2 * λ_nvar * weight * diff[ch]

D_i             = Σ_ch diff[ch]²
remaining_nvar -= weight * D_i
dL_dalpha      += λ_nvar * (T * D_i - remaining_nvar / (1 - alpha)) / T
```

### 5. Normal Un-normalization  (chain rule through normalization)

```
length          = ‖normal‖
dL_dlength      = Σ_i (dL_dnormal_norm[i] * n[i]) / length²
dL_dnormal[i]   = (-dL_dnormal_norm[i] + dL_dlength * n[i]) / length
```

### 6. Confidence

```
dL_dconfidence[id] += weight * (dL_dconf_pixel / (1 - T_final))
```
*(division by `(1 - T_final)` pre-applied in prep)*

### 7. Distortion (DETACH_ALPHA = true)

```
mapped_t        = (far * t - far * near) / ((far - near) * t)
dmax_t_dd       = far * near / ((far - near) * t²)

dL_dmax_t       = 2 * weight * (mapped_t * final_A - final_D) * λ_dist * dmax_t_dd
dL_dt          += dL_dmax_t
```

### 8. Depth at max_contributor  (`current_contributor == max_contributor`)

```
dL_dt += dL_dmax_depth
```

**Case A — plain depth** (`test_T >= 0.5`, or EXACT_DEPTH disabled):

*Gradient flows through common path below (t = -BB/(2*AA)), nothing extra here.*

**Case B — exact depth** (`test_T < 0.5`, EXACT_DEPTH enabled):

```
mu              = -BB / (2*AA)
Fp              = -0.5 / (T * w) + 1 / w     =  (T - 0.5) / (T * w)
log_fp          = log(Fp)
inner           = -2 * log_fp / AA
offset          = sqrt(max(ε, inner))
t_star          = mu - offset

dt_star/dA      = BB / (2*AA²)    +    offset / (2*AA)    [mu + (-offset) terms]
dt_star/dB      = -1 / (2*AA)
dt_star/dw      = -1 / (AA * w * offset)

dL_dA          += dL_dmax_depth * dt_star/dA
dL_dB          += dL_dmax_depth * dt_star/dB
dL_do          += dL_dmax_depth * dt_star/dw          ← goes to dL_dopacity[id]

dL_dt          -= dL_dmax_depth          ← remove from common path to avoid double-count
```

*Store for deferred gradient:*
```
depth_global_id = global_id
dt_dA, dt_dB, dt_dw  ← as above (for opacity-field path, see fin_function)
```

### 9. Opacity Field Loss  (ALPHA_OFFSET regularization → 0.5)

**Tail gaussians** (`t > max_depth`, i.e., peak past the surface):

```
min_value_at_depth = AA * depth² + BB * depth + CC
p                  = min(0, -0.5 * min_value_at_depth)
alpha_point        = min(0.99, w * exp(p))

dL_dalpha_point    = (T_opa - (opacity_final - opacity) / (1 - alpha_point)) * dL_dopacity

dL_do             += dL_dalpha_point * exp(p)          ← opacity param
dL_dA             += dL_dalpha_point * w*exp(p) * (-0.5) * depth²
dL_dB             += dL_dalpha_point * w*exp(p) * (-0.5) * depth
dL_dC             += dL_dalpha_point * w*exp(p) * (-0.5)

                                    [accumulated for deferred gradient on depth gaussian:]
blend_data.dL_dt  += dL_dalpha_point * (-alpha_point) * (AA * depth + BB / 2)

opacity           += alpha_point * T_opa
T_opa             *= (1 - alpha_point)
```

**Head gaussians** (`t <= max_depth`, i.e., peak before the surface):

```
alpha_point        = alpha   (same as rendering alpha)
dL_dalpha         += (T_opa - (opacity_final - opacity) / (1 - alpha)) * dL_dopacity

opacity           += alpha_point * T_opa
T_opa             *= (1 - alpha_point)
```

### 10. Extent / Span Loss

```
C_span          = CC - 2 * log(255 * w)
extent          = sqrt(|BB² - 4*AA*C_span| + ε)
FN              = far * near / (far - near)
NDCspan         = FN * 2*AA*extent / BB²

dL_dNDC         = T * λ_ext * FN / (BB² * extent)      [× alpha if include_alpha]

dL_dA          += (2 * (BB² - 6*AA*C_span))            * dL_dNDC
dL_dB          += (16*AA²*C_span - 2*AA*BB²) / BB      * dL_dNDC
dL_dC          += (-4*AA²)                              * dL_dNDC
dL_do          += (8*AA²) / w                           * dL_dNDC    [if not detach]

    [if include_alpha and not detach:]
    remaining_ext  -= T * alpha * NDCspan
    dL_dalpha      += T * (NDCspan - remaining_ext / test_T) * λ_ext
    [else:]
    remaining_ext  -= T * NDCspan
    dL_dalpha      += remaining_ext / (1 - alpha) * λ_ext
```

### 11. Occupation / Occupation²

```
peak_exp        = -0.5 * (CC - BB² / (4*AA))
geom            = exp(peak_exp)

dL_dgeom        = dL_docc + dL_docc2 * 2 * geom
dL_dexp         = dL_dgeom * geom

dL_dC          += dL_dexp * (-0.5)
dL_dB          += dL_dexp * 0.25 * (BB / AA)
dL_dA          += dL_dexp * (-0.125) * (BB / AA)²
```

### 12. Background

```
bg_dot          = Σ_ch bg[ch] * dL_dpixel[ch]
dL_dalpha      += (-T_final / (1 - alpha)) * bg_dot
```

---

## Accumulate and Write

### dL_dalpha → dL_dG → output gradients

```
dL_dalpha      *= T                     ← scale by transmittance
dL_dG           = w * dL_dalpha
dL_dopacity[id] += G * dL_dalpha + dL_do

gdx             = G * (mean2D.x - px)
gdy             = G * (mean2D.y - py)
dG_ddelx        = -gdx * conic.x - gdy * conic.y
dG_ddely        = -gdy * conic.z - gdx * conic.y
dL_dmean2D.x   += dL_dG * dG_ddelx * (0.5 * W)
dL_dmean2D.y   += dL_dG * dG_ddely * (0.5 * H)
```

### dL_dA, dL_dB from G (peak evaluation)

```
min_value_peak  = -(BB²)/(4*AA) + CC
dL_dmin_value   = (-0.5) * dL_dG * G            ← G = exp(-0.5 * min_value_peak)

dL_dA          += dL_dmin_value * (BB/AA)² / 4
dL_dB          += dL_dmin_value * (-BB / (2*AA))
dL_dC          += dL_dmin_value
```

### dL_dt (distortion + depth) → dL_dA, dL_dB  [common path, plain t = -BB/(2*AA)]

```
dL_dA          += dL_dt * BB / (2*AA²)
dL_dB          += dL_dt * (-1 / (2*AA))
```

*(For the max_contributor with exact depth, dL_dmax_depth was already removed from dL_dt above.)*

### dL_dA, dL_dB, dL_dC → view2gaussian (chain through normal)

```
dL_dnormal[0]  += dL_dA * ray.x
dL_dnormal[1]  += dL_dA * ray.y
dL_dnormal[2]  += dL_dA

view2gaussian layout  [10 floats per gaussian]:
  v[0]  = M[0,0]   → dL_dv[0] += dL_dnormal[0] * ray.x
  v[1]  = M[0,1]   → dL_dv[1] += dL_dnormal[0] * ray.y + dL_dnormal[1] * ray.x
  v[2]  = M[0,2]   → dL_dv[2] += dL_dnormal[0] + dL_dnormal[2] * ray.x
  v[3]  = M[1,1]   → dL_dv[3] += dL_dnormal[1] * ray.y
  v[4]  = M[1,2]   → dL_dv[4] += dL_dnormal[1] + dL_dnormal[2] * ray.y
  v[5]  = M[2,2]   → dL_dv[5] += dL_dnormal[2]
  v[6]  = t_x      → dL_dv[6] += dL_dB * 2 * ray.x
  v[7]  = t_y      → dL_dv[7] += dL_dB * 2 * ray.y
  v[8]  = t_z      → dL_dv[8] += dL_dB * 2
  v[9]  = C (bias) → dL_dv[9] += dL_dC
```

---

## Deferred Gradient (fin_function)

Accumulates `blend_data.dL_dt` = dL/d(max_depth) from all depth-clipped gaussians.
Applies to `depth_global_id` (the gaussian that defined max_depth).

```
if depth_global_id != -1 and blend_data.dL_dt != 0:

    dL_dA           = dt_dA * blend_data.dL_dt
    dL_dB           = dt_dB * blend_data.dL_dt
    dL_dopacity[depth_global_id] += dt_dw * blend_data.dL_dt

    dL_dnormal[0]   = dL_dA * ray.x
    dL_dnormal[1]   = dL_dA * ray.y
    dL_dnormal[2]   = dL_dA

    → atomicAdd to dL_dview2gaussian[depth_global_id] as above
```

Where `dt_dA, dt_dB, dt_dw` were stored at max_contributor time:

| Case | dt_dA | dt_dB | dt_dw |
|------|-------|-------|-------|
| plain (`test_T >= 0.5`) | `BB/(2*AA²)` | `-1/(2*AA)` | `0` |
| exact (`test_T < 0.5`)  | `BB/(2*AA²) + offset/(2*AA)` | `-1/(2*AA)` | `-1/(AA*w*offset)` |

---

## Opacity Forward (_opacity pass) — what it produces for backward

| Written to | Value |
|-----------|-------|
| `ALPHA_OFFSET` | `opacity_head + T_opa_head * opacity_tail` |
| `final_Ts[pix + 3*H*W]` | `T_opa_head * T_opa_tail` |

`opacity_tail` and `T_opa_tail` come from the _forward pass.
`opacity_head` re-evaluates all gaussians up to `max_contributor`, clipped at `max_depth`.
