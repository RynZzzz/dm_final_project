import { useState } from 'react'

const LABEL_SHORT = {
  'expresses confusion or difficulty understanding the content': 'Confusion',
  'asks a clarifying question about the content':               'Clarifying',
  'expresses frustration with the explanation':                 'Frustration',
  'gives positive feedback or appreciation':                    'Positive',
  'is neutral or off-topic':                                    'Neutral',
}

const LABEL_CLASS = {
  'expresses confusion or difficulty understanding the content': 'label-confusion',
  'asks a clarifying question about the content':               'label-clarifying',
  'expresses frustration with the explanation':                 'label-frustration',
  'gives positive feedback or appreciation':                    'label-positive',
  'is neutral or off-topic':                                    'label-neutral',
}

function fmtTs(s) {
  const total = Math.floor(s)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const sec = total % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  return `${m}:${String(sec).padStart(2, '0')}`
}

export default function ConceptCard({ report }) {
  const [open, setOpen] = useState(false)
  const tsRange = `${fmtTs(report.timestamp_start)} – ${fmtTs(report.timestamp_end)}`

  return (
    <div className="concept-card">
      <button className="concept-header" onClick={() => setOpen(!open)}>
        <span className="concept-name">{report.concept_name}</span>
        <span className="concept-meta">{tsRange}</span>
        <span className="concept-count">{report.matched_comment_count} comment{report.matched_comment_count !== 1 ? 's' : ''}</span>
        <span className="concept-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="concept-body">
          <p className="concept-explanation">{report.explanation}</p>

          {report.keywords?.length > 0 && (
            <div className="concept-keywords">
              <span className="keyword-label">Keywords:</span>
              {report.keywords.map((kw, i) => (
                <span key={i} className="keyword-tag">{kw}</span>
              ))}
            </div>
          )}

          {report.synthesis && (
            <div className="synthesis-box">
              <p className="synthesis-label">Summary of viewer feedback</p>
              <p style={{ fontSize: '0.88rem', lineHeight: 1.6 }}>{report.synthesis.summary}</p>
              {report.synthesis.quoted_evidence?.length > 0 && (
                <>
                  <p className="synthesis-label" style={{ marginTop: '0.6rem' }}>Representative quotes</p>
                  {report.synthesis.quoted_evidence.map((q, i) => (
                    <blockquote key={i} className="quote">{q}</blockquote>
                  ))}
                </>
              )}
            </div>
          )}

          {report.matched_comments?.length > 0 && (
            <div className="comments-list">
              <p className="synthesis-label">All matched comments</p>
              {report.matched_comments.map((mc, i) => {
                const tsStr = mc.timestamps_in_text?.length
                  ? mc.timestamps_in_text.map(fmtTs).join(', ')
                  : null
                const labelShort = LABEL_SHORT[mc.predicted_label] || mc.predicted_label
                const labelCls   = LABEL_CLASS[mc.predicted_label]  || 'label-neutral'
                return (
                  <div key={i} className="comment-row">
                    <span className="comment-ts">
                      {tsStr ?? <em style={{ color: '#b2bec3', fontStyle: 'normal' }}>—</em>}
                    </span>
                    <span className="comment-text">{mc.text}</span>
                    <span className="comment-meta">
                      <span className={`label-badge ${labelCls}`}>{labelShort}</span>
                      {' '}conf {mc.classification_confidence.toFixed(2)} · sim {mc.similarity_score.toFixed(2)}
                      {mc.has_timestamp === false && (
                        <span className="content-match-badge">content match</span>
                      )}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
