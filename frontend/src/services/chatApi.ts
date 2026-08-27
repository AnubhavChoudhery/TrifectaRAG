/**
 * Chat API layer — swap `sendChatMessage` with a real backend call later.
 *
 * Backend integration point:
 *   POST /api/chat
 *   Body: { conversation_id, mode, message, attachments? }
 *   Response: { content: string, image_preview?: { url, caption? } }
 */
import type { ChatApiResponse, ChatMode, MessageAttachment } from '../types/chat'

const MOCK_DELAY_MS = 900

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function normalizeMathInput(message: string): string {
  return message
    .toLowerCase()
    .replace(/\*\*/g, '^')
    .replace(/−/g, '-')
    .replace(/\s+/g, '')
}

function formatNumber(n: number): string {
  if (Number.isInteger(n)) return String(n)
  return Number(n.toFixed(6)).toString()
}

function formatSignedNumber(n: number): string {
  return n < 0 ? `- ${formatNumber(Math.abs(n))}` : `+ ${formatNumber(n)}`
}

function cleanExpression(raw: string): string {
  return raw
    .replace(/^f\(x\)=/i, '')
    .replace(/^y=/i, '')
    .replace(/\*\*/g, '^')
    .replace(/−/g, '-')
    .replace(/\s+/g, '')
}

function extractExpression(message: string, keywords: string[]): string | null {
  const compact = cleanExpression(message.toLowerCase())
  for (const keyword of keywords) {
    const index = compact.indexOf(keyword)
    if (index >= 0) {
      const expr = compact.slice(index + keyword.length).replace(/dx$/, '').replace(/=0$/, '')
      return expr || null
    }
  }
  return null
}

type PolynomialTerm = {
  coeff: number
  power: number
}

function parsePolynomial(expr: string): PolynomialTerm[] | null {
  const normalized = cleanExpression(expr).replace(/-/g, '+-')
  const pieces = normalized.split('+').filter(Boolean)
  if (!pieces.length) return null

  const terms: PolynomialTerm[] = []
  for (const piece of pieces) {
    const term = piece.replace(/\*/g, '')
    if (term.includes('x')) {
      const [coeffRaw, powerRaw] = term.split('x')
      let coeff = 1
      if (coeffRaw === '-') coeff = -1
      else if (coeffRaw && coeffRaw !== '+') coeff = Number(coeffRaw)

      const power = powerRaw?.startsWith('^') ? Number(powerRaw.slice(1)) : 1
      if (!Number.isFinite(coeff) || !Number.isFinite(power)) return null
      terms.push({ coeff, power })
    } else {
      const coeff = Number(term)
      if (!Number.isFinite(coeff)) return null
      terms.push({ coeff, power: 0 })
    }
  }

  return terms
}

function formatPolynomialTerm(coeff: number, power: number, isFirst: boolean): string {
  if (coeff === 0) return ''

  const sign = coeff < 0 ? '-' : '+'
  const absCoeff = Math.abs(coeff)
  const prefix = isFirst ? (coeff < 0 ? '-' : '') : ` ${sign} `

  if (power === 0) return `${prefix}${formatNumber(absCoeff)}`

  const coeffText = absCoeff === 1 ? '' : formatNumber(absCoeff)
  const variable = power === 1 ? 'x' : `x^{${power}}`
  return `${prefix}${coeffText}${variable}`
}

function formatPolynomial(terms: PolynomialTerm[]): string {
  const ordered = terms
    .filter((t) => Math.abs(t.coeff) > 1e-12)
    .sort((a, b) => b.power - a.power)

  if (!ordered.length) return '0'

  let first = true
  const text = ordered
    .map((term) => {
      const rendered = formatPolynomialTerm(term.coeff, term.power, first)
      if (rendered) first = false
      return rendered
    })
    .filter(Boolean)
    .join('')

  return text || '0'
}

function solveQuadratic(message: string): string | null {
  const normalized = normalizeMathInput(message)

  const match = normalized.match(/([+-]?\d*)x\^2([+-]\d*)x([+-]\d+)=0/)
  if (!match) return null

  const parseCoeff = (raw: string, fallback = 1) => {
    if (raw === '' || raw === '+') return fallback
    if (raw === '-') return -fallback
    return Number(raw)
  }

  const a = parseCoeff(match[1], 1)
  const b = parseCoeff(match[2], 1)
  const c = Number(match[3])
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || a === 0) return null

  const discriminant = b * b - 4 * a * c
  const sqrtD = Math.sqrt(discriminant)
  const rootsAreIntegers = discriminant >= 0 && Number.isInteger(sqrtD)
  const r1 = (-b + sqrtD) / (2 * a)
  const r2 = (-b - sqrtD) / (2 * a)

  const factorLine =
    rootsAreIntegers && Number.isInteger(r1) && Number.isInteger(r2)
      ? `\nWe can also factor it:\n\n$$\nx^2 ${b < 0 ? '-' : '+'} ${Math.abs(b)}x ${c < 0 ? '-' : '+'} ${Math.abs(c)} = (x ${-r1 < 0 ? '-' : '+'} ${Math.abs(-r1)})(x ${-r2 < 0 ? '-' : '+'} ${Math.abs(-r2)})\n$$\n`
      : ''

  return `## Step-by-step solution

We need to solve:

$$
${a === 1 ? '' : a}x^2 ${b < 0 ? '-' : '+'} ${Math.abs(b)}x ${c < 0 ? '-' : '+'} ${Math.abs(c)} = 0
$$

Use the quadratic formula:

$$
x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}
$$

Here:

$$
a=${a},\\quad b=${b},\\quad c=${c}
$$

Compute the discriminant:

$$
b^2 - 4ac = (${b})^2 - 4(${a})(${c}) = ${discriminant}
$$

So:

$$
x = \\frac{${-b} \\pm \\sqrt{${discriminant}}}{${2 * a}}
$$
${factorLine}
Therefore:

$$
x = ${r1}${r1 === r2 ? '' : `\\quad \\text{or} \\quad x = ${r2}`}
$$`
}

function solveDerivative(message: string): string | null {
  const expr = extractExpression(message, ['derivativeof', 'differentiate', 'd/dx'])
  if (!expr) return null

  if (/sin\(x\^2\)/.test(expr)) {
    return `## Derivative

Let:

$$
f(x)=\\sin(x^2)
$$

Use the chain rule. If $u=x^2$, then:

$$
\\frac{d}{dx}\\sin(u)=\\cos(u)\\frac{du}{dx}
$$

Since $\\frac{du}{dx}=2x$:

$$
f'(x)=2x\\cos(x^2)
$$`
  }

  const terms = parsePolynomial(expr)
  if (!terms) return null

  const derivative = terms.map((term) => ({
    coeff: term.coeff * term.power,
    power: Math.max(term.power - 1, 0),
  }))

  return `## Derivative

Start with:

$$
f(x)=${formatPolynomial(terms)}
$$

Use the power rule:

$$
\\frac{d}{dx}x^n = nx^{n-1}
$$

Differentiate each term:

$$
f'(x)=${formatPolynomial(derivative)}
$$`
}

function solveIntegral(message: string): string | null {
  const compact = normalizeMathInput(message)

  if (/∫?2xcos\(x\^2\)dx/.test(compact) || /integralof2xcos\(x\^2\)/.test(compact)) {
    return `## Integral

Compute:

$$
\\int 2x\\cos(x^2)\\,dx
$$

Use substitution:

$$
u=x^2,\\quad du=2x\\,dx
$$

So the integral becomes:

$$
\\int \\cos(u)\\,du = \\sin(u)+C
$$

Substitute back:

$$
\\int 2x\\cos(x^2)\\,dx = \\sin(x^2)+C
$$`
  }

  const expr = extractExpression(message, ['integralof', 'integrate', '∫'])
  if (!expr) return null

  const terms = parsePolynomial(expr.replace(/dx$/, ''))
  if (!terms) return null

  const integral = terms.map((term) => ({
    coeff: term.coeff / (term.power + 1),
    power: term.power + 1,
  }))

  return `## Integral

Compute:

$$
\\int ${formatPolynomial(terms)}\\,dx
$$

Use the power rule for integration:

$$
\\int x^n\\,dx = \\frac{x^{n+1}}{n+1}+C
$$

Integrate term by term:

$$
\\int ${formatPolynomial(terms)}\\,dx = ${formatPolynomial(integral)} + C
$$`
}

function solveLinearEquation(message: string): string | null {
  const normalized = normalizeMathInput(message)
  const match = normalized.match(/([+-]?\d*)x([+-]\d+)=([+-]?\d+)/)
  if (!match || normalized.includes('x^2')) return null

  const parseCoeff = (raw: string) => {
    if (raw === '' || raw === '+') return 1
    if (raw === '-') return -1
    return Number(raw)
  }

  const a = parseCoeff(match[1])
  const b = Number(match[2])
  const c = Number(match[3])
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || a === 0) return null

  const x = (c - b) / a
  return `## Linear equation

Solve:

$$
${a === 1 ? '' : a}x ${formatSignedNumber(b)} = ${c}
$$

Move the constant term:

$$
${a === 1 ? '' : a}x = ${c} ${formatSignedNumber(-b)} = ${formatNumber(c - b)}
$$

Divide by ${a}:

$$
x = ${formatNumber(x)}
$$`
}

function solveMath(message: string): string | null {
  return solveQuadratic(message) ?? solveDerivative(message) ?? solveIntegral(message) ?? solveLinearEquation(message)
}

function mockResponse(mode: ChatMode, message: string): ChatApiResponse {
  const q = message.trim()

  switch (mode) {
    case 'math':
      {
        const solved = solveMath(q)
        if (solved) {
          return { content: solved }
        }
      }
      return {
        content: `I can solve many standard algebra and calculus questions directly. Try writing the expression clearly, for example:

- \`Solve x^2 - 5x + 6 = 0 step by step\`
- \`Differentiate x^3 - 4x^2 + 2x - 7\`
- \`Integrate 3x^2 - 4x + 1\`
- \`Find the derivative of sin(x^2)\`
- \`Integrate 2x cos(x^2) dx\`

$$
\\int x^n\\,dx = \\frac{x^{n+1}}{n+1}+C
$$

$$
\\frac{d}{dx}x^n = nx^{n-1}
$$
`,
      }

    case 'code':
      return {
        content: `Here's a clean Python implementation for your request:

\`\`\`python
def binary_search(arr: list[int], target: int) -> int:
    """Return index of target in sorted arr, or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
\`\`\`

**Complexity:** $O(\\log n)$ time, $O(1)$ space.

| Approach | Time | Space |
|----------|------|-------|
| Linear scan | $O(n)$ | $O(1)$ |
| Binary search | $O(\\log n)$ | $O(1)$ |

Want me to add tests or convert this to TypeScript?`,
      }

    case 'study':
      return {
        content: `Upload a PDF first, then ask your study question. Study Tutor answers from the indexed document only, so it does not invent unrelated notes.

Once the upload says **PDF ready**, ask questions like:

- \`What is the difference between a double salt and a complex?\`
- \`Explain this topic in simple words.\`
- \`List important formulas from this chapter.\``,
      }

    case 'research':
      return {
        content: `## Literature Summary

Based on your question about **"${q.slice(0, 60)}${q.length > 60 ? '…' : ''}"**, here is a structured overview:

| Study | Method | Key Finding |
|-------|--------|-------------|
| Smith et al. (2022) | RCT, n=240 | +18% retention vs. control |
| Chen & Patel (2023) | Meta-analysis | Effect size $d = 0.42$ |
| Nguyen (2024) | Longitudinal | Gains persist at 6 months |

### Takeaways

1. Evidence supports moderate positive effects.
2. Heterogeneity remains across populations.
3. More replication studies are needed.

> *Note: Connect your RAG backend to replace this with real retrieved citations.*`,
        imagePreview: {
          url: 'https://images.unsplash.com/photo-1532094349884-543bc11b234d?w=640&q=80',
          caption: 'Example figure reference — replace with retrieved document images',
          alt: 'Scientific laboratory',
        },
      }

    default:
      return {
        content: `Thanks for your message! I'm here to help.

You asked: *"${q}"*

Here's what I can do in **General Chat** mode:

- Answer questions clearly
- Render **Markdown**, \`inline code\`, and [links](https://example.com)
- Show math like $\\int_0^1 x^2\\,dx = \\frac{1}{3}$

> Choose a specialized mode from the header for math, code, study, or research workflows.`,
      }
  }
}

/** Replace this function body with fetch('/api/chat', …) when wiring the backend. */
export async function sendChatMessage(
  mode: ChatMode,
  message: string,
  _attachments?: MessageAttachment[],
): Promise<ChatApiResponse> {
  await delay(MOCK_DELAY_MS + Math.random() * 400)
  return mockResponse(mode, message)
}
