import { useEffect, useRef, useState } from 'react'
import { AlertCircle, CheckCircle2, Clock, FileText, Loader2, Upload } from 'lucide-react'
import { getIngestStatus, uploadPdf } from '../services/api.ts'

const POLL_INTERVAL_MS = 2000

export default function UploadCard({ onUploaded }) {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [taskId, setTaskId] = useState(null)
  const pollRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (status !== 'processing' || !taskId) return undefined

    pollRef.current = setInterval(async () => {
      try {
        const data = await getIngestStatus(taskId)
        if (data.status === 'done') {
          clearInterval(pollRef.current)
          setResult(data)
          setStatus('done')
          onUploaded?.(data)
        } else if (data.status === 'error') {
          clearInterval(pollRef.current)
          setErrorMsg(data.error || 'Ingestion failed.')
          setStatus('error')
        }
      } catch (err) {
        setErrorMsg(err.message || 'Could not read ingestion status.')
        setStatus('error')
        clearInterval(pollRef.current)
      }
    }, POLL_INTERVAL_MS)

    return () => clearInterval(pollRef.current)
  }, [status, taskId, onUploaded])

  const busy = status === 'uploading' || status === 'processing'
  const stats = result?.stats

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0]
    if (selected && selected.type === 'application/pdf') {
      setFile(selected)
      setStatus('idle')
      setErrorMsg('')
      setResult(null)
      return
    }

    setFile(null)
    setErrorMsg('Select a PDF file before ingestion.')
    setStatus('error')
  }

  const handleUpload = async () => {
    if (!file) return
    setStatus('uploading')
    setErrorMsg('')
    setResult(null)
    setTaskId(null)

    try {
      const data = await uploadPdf(file)
      setTaskId(data.task_id)
      setStatus('processing')
    } catch (err) {
      setErrorMsg(err.message || 'Upload failed. Check that the backend is running.')
      setStatus('error')
    }
  }

  return (
    <section className="panel">
      <div className="section-heading">
        <div className="icon-tile bg-teal-700 text-white">
          <FileText size={18} />
        </div>
        <div>
          <h2>Document Ingestion</h2>
          <p>Upload a PDF and build the retrieval index</p>
        </div>
      </div>

      <button
        type="button"
        onClick={() => !busy && inputRef.current?.click()}
        className="drop-zone"
        disabled={busy}
      >
        <Upload size={24} className="text-teal-700" />
        <span>{file ? file.name : 'Select a PDF textbook or notes file'}</span>
        <small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'Stored under uploaded_pdfs and indexed in the background'}</small>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={handleFileChange}
        />
      </button>

      <button
        type="button"
        onClick={handleUpload}
        disabled={!file || busy}
        className="primary-button mt-4 w-full"
      >
        {status === 'uploading' || status === 'processing' ? (
          <>
            <Loader2 size={16} className="animate-spin" />
            {status === 'uploading' ? 'Uploading' : 'Indexing'}
          </>
        ) : (
          <>
            <Upload size={16} />
            Start ingestion
          </>
        )}
      </button>

      {status === 'processing' && (
        <div className="status-box border-teal-200 bg-teal-50 text-teal-900">
          <Clock size={16} />
          <div>
            <strong>Indexing in progress</strong>
            <span>Extracting text, figures, metadata, embeddings, and graph edges.</span>
          </div>
        </div>
      )}

      {status === 'done' && stats && (
        <div className="status-box border-emerald-200 bg-emerald-50 text-emerald-900">
          <CheckCircle2 size={16} />
          <div className="w-full">
            <strong>{result.filename} is indexed</strong>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {[
                ['Pages', stats.pages],
                ['Text', stats.text_chunks],
                ['Figures', stats.images],
                ['Edges', stats.kg_edges],
              ].map(([label, value]) => (
                <div key={label} className="mini-stat">
                  <span>{value}</span>
                  <small>{label}</small>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {status === 'error' && errorMsg && (
        <div className="status-box border-red-200 bg-red-50 text-red-900">
          <AlertCircle size={16} />
          <span>{errorMsg}</span>
        </div>
      )}
    </section>
  )
}
