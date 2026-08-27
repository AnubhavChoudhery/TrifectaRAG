import { useEffect, useState } from 'react'
import { Loader2, Search, Send, SlidersHorizontal } from 'lucide-react'
import { askQuestion } from '../services/api.ts'

const TOP_K_OPTIONS = [3, 5, 8, 10]

export default function QuestionCard({ onResult, prompt, onPromptConsumed }) {
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!prompt) return
    setQuestion(prompt)
    onPromptConsumed?.()
  }, [prompt, onPromptConsumed])

  const handleAsk = async () => {
    const q = question.trim()
    if (!q) return
    setLoading(true)
    setError('')
    onResult(null)

    try {
      const data = await askQuestion(q, topK)
      onResult(data)
    } catch (err) {
      setError(err.message || 'Retrieval failed. Check that the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleAsk()
    }
  }

  return (
    <section className="panel">
      <div className="section-heading">
        <div className="icon-tile bg-slate-900 text-white">
          <Search size={18} />
        </div>
        <div>
          <h2>Ask</h2>
          <p>Search semantic, lexical, and graph context together</p>
        </div>
      </div>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={7}
        placeholder="Ask for a concept, formula, derivation, source page, or related figure"
        className="query-box"
      />

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <SlidersHorizontal size={15} className="text-slate-500" />
          <span className="text-xs font-medium text-slate-500">top_k</span>
          <div className="segmented-control">
            {TOP_K_OPTIONS.map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setTopK(k)}
                className={topK === k ? 'active' : ''}
              >
                {k}
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={handleAsk}
          disabled={loading || !question.trim()}
          className="primary-button"
        >
          {loading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Searching
            </>
          ) : (
            <>
              <Send size={16} />
              Search
            </>
          )}
        </button>
      </div>

      {error && <div className="error-line">{error}</div>}
    </section>
  )
}
