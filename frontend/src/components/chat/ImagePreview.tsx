import { ExternalLink, ImageIcon } from 'lucide-react'
import type { ImagePreview } from '../../types/chat'

type ImagePreviewProps = {
  preview: ImagePreview
  compact?: boolean
}

export default function ImagePreviewCard({ preview, compact = false }: ImagePreviewProps) {
  return (
    <figure
      className={`group overflow-hidden rounded-xl border border-chat-border bg-chat-muted/40 ${
        compact ? 'max-w-xs' : 'max-w-md'
      }`}
    >
      <div className="relative">
        <img
          src={preview.url}
          alt={preview.alt ?? preview.caption ?? 'Referenced image'}
          className="max-h-72 w-full object-cover transition group-hover:opacity-95"
          loading="lazy"
        />
        <a
          href={preview.url}
          target="_blank"
          rel="noopener noreferrer"
          className="absolute right-2 top-2 rounded-lg bg-black/50 p-1.5 text-white opacity-0 transition group-hover:opacity-100"
          aria-label="Open image in new tab"
        >
          <ExternalLink size={14} />
        </a>
      </div>
      {(preview.caption || preview.alt) && (
        <figcaption className="flex items-start gap-2 px-3 py-2.5 text-xs leading-5 text-chat-muted-fg">
          <ImageIcon size={14} className="mt-0.5 shrink-0" />
          <span>{preview.caption ?? preview.alt}</span>
        </figcaption>
      )}
    </figure>
  )
}
