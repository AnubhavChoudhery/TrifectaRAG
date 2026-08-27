import { useState } from 'react'
import { ChevronDown, ChevronUp, FileText, Hash, Image, MapPin, Star } from 'lucide-react'

function ScoreBadge({ score }) {
  return (
    <span className="source-badge bg-amber-50 text-amber-800">
      <Star size={11} />
      {Number(score ?? 0).toFixed(4)}
    </span>
  )
}

function ModalityBadge({ modality }) {
  const isImage = modality === 'IMAGE'
  return (
    <span className={`source-badge ${isImage ? 'bg-fuchsia-50 text-fuchsia-800' : 'bg-sky-50 text-sky-800'}`}>
      {isImage ? <Image size={11} /> : <FileText size={11} />}
      {modality || 'TEXT'}
    </span>
  )
}

function SourceCard({ source }) {
  const imageUrl = source.image_path
    ? `/image?path=${encodeURIComponent(source.image_path)}`
    : null

  return (
    <article className="source-card">
      <div className="flex flex-wrap items-center gap-2">
        <span className="source-badge bg-slate-100 text-slate-700">
          <Hash size={11} />
          {source.global_id}
        </span>
        <ModalityBadge modality={source.modality} />
        <ScoreBadge score={source.score} />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
        <span className="inline-flex min-w-0 items-center gap-1 font-medium text-slate-700">
          <MapPin size={12} />
          <span className="truncate">{source.source || 'Unknown source'}</span>
        </span>
        <span>page {source.page || '?'}</span>
      </div>

      {source.text_preview && (
        <p className="mt-3 line-clamp-5 text-sm leading-6 text-slate-700">
          {source.text_preview}
        </p>
      )}

      {imageUrl && (
        <div className="mt-3 rounded-lg border border-slate-200 bg-white p-2">
          <img
            src={imageUrl}
            alt={`Figure from ${source.source || 'source'} page ${source.page || '?'}`}
            className="max-h-56 w-full rounded object-contain"
            loading="lazy"
          />
        </div>
      )}
    </article>
  )
}

export default function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(true)

  if (!sources || sources.length === 0) return null

  return (
    <section className="panel">
      <button type="button" onClick={() => setOpen((v) => !v)} className="section-toggle">
        <span className="section-heading mb-0">
          <span className="icon-tile bg-slate-900 text-white">
            <FileText size={18} />
          </span>
          <span>
            <h2>Retrieved Sources</h2>
            <p>{sources.length} ranked global_id chunks returned by the engine</p>
          </span>
        </span>
        <span className="inline-flex items-center gap-1 text-sm font-medium text-teal-700">
          {open ? 'Hide' : 'Show'}
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>

      {open && (
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {sources.map((src, i) => (
            <SourceCard key={src.global_id ?? i} source={src} />
          ))}
        </div>
      )}
    </section>
  )
}
