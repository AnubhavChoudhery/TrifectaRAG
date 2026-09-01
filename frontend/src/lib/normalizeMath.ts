/**
 * Convert common model math dialects into remark-math / KaTeX delimiters.
 * qwen2.5 often emits \(...\), \[...\], or [ equation ] instead of $ / $$.
 */
export function normalizeMath(content: string): string {
  if (!content || (!content.includes('\\') && !content.includes('_') && !content.includes('['))) {
    return content
  }

  const parts = content.split(/(```[\s\S]*?```)/g)
  return parts
    .map((part, index) => (index % 2 === 1 ? part : normalizeMathSegment(part)))
    .join('')
}

function normalizeMathSegment(segment: string): string {
  let text = segment
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_m, inner: string) => `\n$$\n${inner.trim()}\n$$\n`)
  text = text.replace(/\\\((.+?)\\\)/g, (_m, inner: string) => `$${inner.trim()}$`)
  text = text.replace(/\[((?:[^[\]\n]|\\\[|\\\])+)](?!\()/g, (full, inner: string) => {
    const body = String(inner).trim()
    if (!looksLikeLatex(body)) return full
    return body.length > 20 || /\\\\|\\frac|\\sum|\\int/.test(body) ? `$$${body}$$` : `$${body}$`
  })
  return text
}

function looksLikeLatex(body: string): boolean {
  return /\\[a-zA-Z]+|[_^]|\\frac|\\in|\\left|\\right|\\cdot|\\times/.test(body)
}
