/**
 * Prepare model / PDF-excerpt markdown for remark-math + KaTeX.
 *
 * qwen2.5 often emits \(...\), \[...\], escaped \$, or $$$ after we
 * wrap already-delimited intervals. PDF excerpts also glue words together.
 */

const GLUED_WORDS = [
  'defined',
  'because',
  'therefore',
  'provided',
  'interval',
  'iteration',
  'iterations',
  'relaxation',
  'sequence',
  'converges',
  'converge',
  'starting',
  'initial',
  'newton',
  'method',
  'theorem',
  'lemma',
  'where',
  'when',
  'which',
  'that',
  'this',
  'with',
  'from',
  'into',
  'onto',
  'then',
  'than',
  'and',
  'the',
  'for',
  'not',
]

export function normalizeMath(content: string): string {
  if (!content) return content
  const parts = content.split(/(```[\s\S]*?```)/g)
  return parts
    .map((part, index) => (index % 2 === 1 ? part : normalizeMathSegment(part)))
    .join('')
}

function normalizeMathSegment(segment: string): string {
  let text = restorePdfSpaces(segment)
  text = text.replace(/\\\$/g, '$')
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_m, inner: string) => `\n$$\n${inner.trim()}\n$$\n`)
  text = text.replace(/\\\((.+?)\\\)/g, (_m, inner: string) => `$${inner.trim()}$`)
  text = collapseDollars(text)
  text = mapOutsideMath(text, wrapIntervals)
  text = mapOutsideMath(text, wrapBareLatex)
  text = collapseDollars(text)
  text = text.replace(/\$([a-z]{2,})/g, '$ $1')
  text = text.replace(/([a-z,;:])(\$(?:\\|[A-Za-z]))/g, '$1 $2')
  text = mapOutsideMath(text, restorePdfSpaces)
  return text
}

function collapseDollars(text: string): string {
  return text.replace(/\${3,}/g, '$$')
}

function mapOutsideMath(text: string, fn: (plain: string) => string): string {
  const chunks = text.split(/(\$\$[\s\S]*?\$\$|\$[^$\n]*\$)/g)
  return chunks
    .map((chunk) => {
      if (chunk.startsWith('$$') || (chunk.startsWith('$') && chunk.endsWith('$') && chunk.length >= 2)) {
        return chunk
      }
      return fn(chunk)
    })
    .join('')
}

function wrapIntervals(plain: string): string {
  return plain.replace(/\[((?:\\[a-zA-Z]+|[^[\]\n])+)\](?!\()/g, (full, inner: string) => {
    if (!looksLikeLatex(inner)) return full
    return `$${full}$`
  })
}

function wrapBareLatex(plain: string): string {
  let text = plain.replace(/\\[a-zA-Z]+(?:_\{[^}]+\})?/g, (cmd) => `$${cmd}$`)
  text = text.replace(/\b([A-Za-z]'?)_(\{[^}]+\}|\d+)(?=[A-Za-z]|\b)/g, (_m, name: string, sub: string) => {
    return `$${name}_${sub}$`
  })
  text = text.replace(/\b([A-Za-z]'?)_([A-Za-z])\b/g, (_m, name: string, sub: string) => {
    return `$${name}_${sub}$`
  })
  return text
}

function looksLikeLatex(body: string): boolean {
  return /\\[a-zA-Z]+|[_^]|\\frac|\\in|\\left|\\right|\\cdot|\\times|\\le|\\ge|\\xi|\\delta/.test(body)
}

function restorePdfSpaces(text: string): string {
  let out = text.replace(/([)\],.:;])([A-Za-z\\])/g, '$1 $2')
  out = out.replace(/([a-z])([A-Z])/g, '$1 $2')
  out = out.replace(/bythe(?=[A-Za-z]|$)/gi, 'by the')
  const words = [...GLUED_WORDS].sort((a, b) => b.length - a.length)
  for (const word of words) {
    const re = new RegExp(`(?<=[a-z0-9)\\]$\\s])(${word})(?=[a-z(\\\\$])`, 'gi')
    out = out.replace(re, (full, matched: string, offset: number, whole: string) => {
      const before = whole.slice(0, offset)
      const prefix = (before.match(/[a-z]+$/i) || [''])[0].toLowerCase()
      const after = whole.slice(offset + full.length)
      const letters = (after.match(/^[a-z]+/) || [''])[0].toLowerCase()
      const combined = `${matched}${letters}`.toLowerCase()
      if (prefix && prefix.length > 1 && !GLUED_WORDS.includes(prefix)) return full
      if (GLUED_WORDS.includes(combined)) return full
      if (words.some((other) => other.length > matched.length && combined.startsWith(other))) {
        return full
      }
      return ` ${matched} `
    })
  }
  return out.replace(/[ \t]{2,}/g, ' ')
}
