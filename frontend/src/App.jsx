import { useState, useEffect } from 'react'
import SentimentChart from './components/SentimentChart'
import ClassificationChart from './components/ClassificationChart'
import ConceptCard from './components/ConceptCard'
import TimelineChart from './components/TimelineChart'
import ConceptMatchChart from './components/ConceptMatchChart'
import ConceptLabelChart from './components/ConceptLabelChart'
import TroubleDensityChart from './components/TroubleDensityChart'
import TroubleMatchDonut from './components/TroubleMatchDonut'

const POLL_MS = 2000

const STAGE_LABELS = {
  ingest:           'Fetching video & comments',
  transcribe:       'Transcribing audio',
  extract_concepts: 'Extracting concepts',
  classify:         'Classifying comments',
  match:            'Matching comments to concepts',
  synthesize:       'Synthesizing feedback',
}

function fmtTs(s) {
  const total = Math.floor(s)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const sec = total % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  return `${m}:${String(sec).padStart(2, '0')}`
}

export default function App() {
  const [url, setUrl]               = useState('')
  const [jobId, setJobId]           = useState(null)
  const [jobStatus, setJobStatus]   = useState(null)
  const [result, setResult]         = useState(null)
  const [error, setError]           = useState(null)
  const [submittedUrl, setSubmittedUrl] = useState(null)

  const reset = () => {
    if (jobId) fetch(`/reset/${jobId}`, { method: 'DELETE' }).catch(() => {})
    setUrl('')
    setJobId(null)
    setJobStatus(null)
    setResult(null)
    setError(null)
    setSubmittedUrl(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!url.trim()) return
    setError(null)
    setSubmittedUrl(url.trim())
    try {
      const resp = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim(), force: false }),
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setJobId(data.job_id)
    } catch (err) {
      setError(`Failed to submit: ${err.message}`)
    }
  }

  useEffect(() => {
    if (!jobId || result) return

    const poll = async () => {
      try {
        const resp = await fetch(`/status/${jobId}`)
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const status = await resp.json()
        setJobStatus(status)

        if (status.status === 'completed') {
          const rResp = await fetch(`/results/${jobId}`)
          if (!rResp.ok) throw new Error(`HTTP ${rResp.status}`)
          setResult(await rResp.json())
        } else if (status.status === 'failed') {
          setError(
            `Analysis failed at stage "${status.stage}": ${status.error_message || 'Unknown error'}`
          )
        }
      } catch (err) {
        setError(`Polling error: ${err.message}`)
      }
    }

    poll()
    const id = setInterval(poll, POLL_MS)
    return () => clearInterval(id)
  }, [jobId, result])

  const isRunning = jobId && !result && jobStatus?.status !== 'failed' && !error

  return (
    <div className="app">

      <header className="app-header">
        <h1>🎓 YouTube Learning Diagnostics</h1>
        <p className="subtitle">
          Analyze educational videos by surfacing the concepts taught and
          the moments viewers expressed difficulty.
        </p>
      </header>

      <section className="input-section">
        <form onSubmit={handleSubmit} className="url-form">
          <input
            className="url-input"
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            disabled={isRunning}
          />
          <button className="btn-primary" type="submit" disabled={isRunning || !url.trim()}>
            Analyze
          </button>
          {(jobId || result) && (
            <button className="btn-secondary" type="button" onClick={reset}>
              Reset
            </button>
          )}
        </form>
      </section>

      {error && <div className="error-box">⚠️ {error}</div>}

      {isRunning && jobStatus && (
        <section className="progress-section">
          <div className="progress-info">
            <span className="progress-stage-name">
              {STAGE_LABELS[jobStatus.stage] || jobStatus.stage || 'Starting…'}
            </span>
            <span className="progress-pct">{jobStatus.progress}%</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${jobStatus.progress}%` }} />
          </div>
          <p className="progress-caption">
            {submittedUrl}<br />
            This typically takes 2–10 minutes depending on video length and comment count.
          </p>
        </section>
      )}

      {result && <Results result={result} />}

    </div>
  )
}

function Results({ result }) {
  const meta    = result.metadata
  const matched = result.trouble_comment_count - result.unmatched_trouble_count

  const withMatches    = result.concept_reports.filter(r => r.matched_comment_count > 0)
  const withoutMatches = result.concept_reports.filter(r => r.matched_comment_count === 0)

  return (
    <div className="results">
      <hr />

      <h2>{meta.title}</h2>
      <div className="metrics-row">
        <Metric label="Channel"          value={meta.uploader} />
        <Metric label="Duration"         value={fmtTs(meta.duration)} />
        <Metric label="Comments fetched" value={result.total_comments} />
        <Metric label="With timestamps"  value={result.timestamped_comments} />
      </div>
      <p className="stats-caption">
        Of <strong>{result.timestamped_comments}</strong> timestamped comments,{' '}
        <strong>{result.trouble_comment_count}</strong> were flagged as expressing difficulty.{' '}
        <strong>{matched}</strong> mapped to specific concepts;{' '}
        <strong>{result.unmatched_trouble_count}</strong> did not.
      </p>

      {(result.sentiment || Object.keys(result.label_distribution || {}).length > 0) && (
        <>
          <hr />
          <div className="charts-row">
            <div className="chart-panel">
              <h3>Overall Sentiment</h3>
              {result.sentiment
                ? <>
                    <SentimentChart sentiment={result.sentiment} />
                    <p className="chart-caption">
                      Sample: {result.sentiment.sample_size} comments ·{' '}
                      <em>{result.sentiment.disclaimer}</em>
                    </p>
                    <CommentSampleList
                      samples={result.sentiment_comment_samples}
                      labelMap={{ positive: 'Positive', neutral: 'Neutral', negative: 'Negative' }}
                      order={['positive', 'neutral', 'negative']}
                      summary="View sample comments by sentiment"
                    />
                  </>
                : <p className="muted">No sentiment data available.</p>
              }
            </div>
            <div className="chart-panel">
              <h3>Comment Classification</h3>
              {Object.keys(result.label_distribution || {}).length > 0
                ? <>
                    <ClassificationChart labelDist={result.label_distribution} />
                    <p className="chart-caption">
                      {Object.values(result.label_distribution).reduce((a, b) => a + b, 0)} timestamped
                      comments classified — <strong>{result.trouble_comment_count}</strong> flagged as trouble.
                    </p>
                    <CommentSampleList
                      samples={result.label_comment_samples}
                      labelMap={{
                        'expresses confusion or difficulty understanding the content': 'Confusion',
                        'asks a clarifying question about the content': 'Clarifying question',
                        'expresses frustration with the explanation': 'Frustration',
                        'gives positive feedback or appreciation': 'Positive feedback',
                        'is neutral or off-topic': 'Neutral / off-topic',
                      }}
                      order={[
                        'expresses confusion or difficulty understanding the content',
                        'asks a clarifying question about the content',
                        'expresses frustration with the explanation',
                        'gives positive feedback or appreciation',
                        'is neutral or off-topic',
                      ]}
                      summary="View sample comments by label"
                    />
                  </>
                : <p className="muted">No classification data available.</p>
              }
            </div>
          </div>
        </>
      )}

      {result.trouble_comment_count > 0 && (
        <>
          <hr />
          <div className="charts-row">
            <div className="chart-panel">
              <h3>Trouble Comment Density</h3>
              <p className="chart-caption" style={{ marginBottom: '0.75rem' }}>
                Trouble comments per time segment — taller bars indicate harder zones.
              </p>
              <TroubleDensityChart
                conceptReports={result.concept_reports}
                unmatchedComments={result.unmatched_trouble_comments}
                duration={result.metadata.duration}
              />
            </div>
            <div className="chart-panel">
              <h3>Concept Mapping Coverage</h3>
              <p className="chart-caption" style={{ marginBottom: '0.75rem' }}>
                Trouble comments matched to a concept vs left unmatched.
              </p>
              <TroubleMatchDonut
                troubleCount={result.trouble_comment_count}
                unmatchedCount={result.unmatched_trouble_count}
              />
            </div>
          </div>
        </>
      )}

      {result.trouble_comment_count > 0 && (
        <>
          <hr />
          <div className="chart-panel" style={{ marginBottom: '1rem' }}>
            <h3>Trouble Comments Along the Timeline</h3>
            <p className="chart-caption" style={{ marginBottom: '0.75rem' }}>
              Each dot is a trouble comment at the timestamp it references.
              Red = confusion · orange = clarifying question · dark red = frustration.
            </p>
            <TimelineChart
              conceptReports={result.concept_reports}
              unmatchedComments={result.unmatched_trouble_comments}
              duration={result.metadata.duration}
            />
          </div>
        </>
      )}

      {result.concept_reports.length > 0 && (
        <>
          <hr />
          <div className="charts-row">
            <div className="chart-panel">
              <h3>All Concepts by Difficulty</h3>
              <p className="chart-caption" style={{ marginBottom: '0.75rem' }}>
                Trouble comments per concept — grey bars have no difficulty reported.
              </p>
              <ConceptMatchChart conceptReports={result.concept_reports} />
            </div>
            <div className="chart-panel">
              <h3>Difficulty Type per Concept</h3>
              <p className="chart-caption" style={{ marginBottom: '0.75rem' }}>
                Breakdown of confusion, clarifying questions, and frustration per concept.
              </p>
              <ConceptLabelChart conceptReports={result.concept_reports} />
            </div>
          </div>
        </>
      )}

      <hr />
      <h3>Concepts & Viewer Feedback</h3>

      {result.concept_reports.length === 0 && (
        <p className="muted">No concepts were extracted from this video.</p>
      )}

      {withMatches.length > 0 && (
        <>
          <p className="section-meta">
            <strong>{withMatches.length}</strong> concept{withMatches.length !== 1 ? 's' : ''} had comments flagged as difficulty:
          </p>
          {withMatches.map((r, i) => <ConceptCard key={i} report={r} />)}
        </>
      )}

      {withoutMatches.length > 0 && (
        <details className="no-match-section">
          <summary>+ {withoutMatches.length} concept(s) with no flagged comments</summary>
          {withoutMatches.map((r, i) => (
            <div key={i} className="no-match-item">
              <strong>{r.concept_name}</strong>
              {' '}({fmtTs(r.timestamp_start)} – {fmtTs(r.timestamp_end)})
              <br />
              <em>{r.explanation}</em>
              <hr className="thin-hr" />
            </div>
          ))}
        </details>
      )}

      {result.unmatched_trouble_comments?.length > 0 && (
        <>
          <hr />
          <details className="unmatched-details">
            <summary>
              {result.unmatched_trouble_comments.length} trouble comment(s) not matched to any concept
            </summary>
            <div className="unmatched-body">
              <p className="muted" style={{ marginBottom: '0.75rem' }}>
                These comments were flagged as expressing difficulty but didn't clearly relate
                to any extracted concept.
              </p>
              {result.unmatched_trouble_comments.map((mc, i) => {
                const tsStr = mc.timestamps_in_text?.length
                  ? mc.timestamps_in_text.map(fmtTs).join(', ')
                  : '—'
                return (
                  <div key={i} className="comment-row">
                    <span className="comment-ts">{tsStr}</span>
                    <span className="comment-text">{mc.text}</span>
                    <span className="comment-meta">best sim {mc.similarity_score.toFixed(2)}</span>
                  </div>
                )
              })}
            </div>
          </details>
        </>
      )}

      {result.disclaimers?.length > 0 && (
        <>
          <hr />
          <h5>About this analysis</h5>
          {result.disclaimers.map((d, i) => (
            <p key={i} className="disclaimer">• {d}</p>
          ))}
        </>
      )}
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
    </div>
  )
}

function CommentSampleList({ samples, labelMap, order, summary }) {
  if (!samples || Object.keys(samples).length === 0) return null
  const hasAny = order.some(k => samples[k]?.length > 0)
  if (!hasAny) return null
  return (
    <details style={{ marginTop: '0.75rem' }}>
      <summary style={{ fontSize: '0.8rem', cursor: 'pointer', color: '#636e72', fontWeight: 600 }}>
        {summary}
      </summary>
      {order.map(key => {
        const list = samples[key]
        if (!list?.length) return null
        return (
          <div key={key} style={{ marginTop: '0.6rem' }}>
            <p style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: '0.25rem', color: '#555', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {labelMap[key] || key} ({list.length})
            </p>
            {list.map((c, i) => (
              <div key={i} className="comment-row">
                <span className="comment-text">{c.text}</span>
                {c.author && <span className="comment-meta">{c.author}</span>}
              </div>
            ))}
          </div>
        )
      })}
    </details>
  )
}
