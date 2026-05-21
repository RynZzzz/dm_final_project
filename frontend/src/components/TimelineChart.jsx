import { useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const LABEL_COLOR = {
  'expresses confusion or difficulty understanding the content': '#e74c3c',
  'asks a clarifying question about the content':               '#f39c12',
  'expresses frustration with the explanation':                 '#c0392b',
}

const LABEL_SHORT = {
  'expresses confusion or difficulty understanding the content': 'Confusion',
  'asks a clarifying question about the content':               'Clarifying question',
  'expresses frustration with the explanation':                 'Frustration',
}

function fmtTs(s) {
  const total = Math.floor(s)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const sec = total % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  return `${m}:${String(sec).padStart(2, '0')}`
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div style={{
      background: '#fff', border: '1px solid #dfe6e9',
      padding: '8px 12px', borderRadius: 6, maxWidth: 260, fontSize: 12,
    }}>
      <div><strong>{fmtTs(d.x)}</strong> — {d.concept}</div>
      <div style={{ color: '#636e72', marginTop: 4 }}>{d.text}</div>
      <div style={{ marginTop: 4, color: LABEL_COLOR[d.label] || '#95a5a6' }}>
        <em>{LABEL_SHORT[d.label] || d.label}</em>
      </div>
    </div>
  )
}

export default function TimelineChart({ conceptReports, unmatchedComments, duration }) {
  const withMatches = conceptReports.filter(r => r.matched_comment_count > 0)
  const hasUnmatched = unmatchedComments?.length > 0

  const { data, conceptNames } = useMemo(() => {
    const names = [
      ...withMatches.map(r => r.concept_name),
      ...(hasUnmatched ? ['Unmatched'] : []),
    ]
    const points = []

    withMatches.forEach((report, ci) => {
      report.matched_comments?.forEach(mc => {
        mc.timestamps_in_text?.forEach(ts => {
          points.push({
            x: ts,
            y: ci,
            label: mc.predicted_label,
            concept: report.concept_name,
            text: mc.text.length > 100 ? mc.text.slice(0, 100) + '…' : mc.text,
          })
        })
      })
    })

    unmatchedComments?.forEach(mc => {
      mc.timestamps_in_text?.forEach(ts => {
        points.push({
          x: ts,
          y: withMatches.length,
          label: mc.predicted_label,
          concept: 'Unmatched',
          text: mc.text.length > 100 ? mc.text.slice(0, 100) + '…' : mc.text,
        })
      })
    })

    return { data: points, conceptNames: names }
  }, [conceptReports, unmatchedComments])

  if (data.length === 0) {
    return <p style={{ color: '#b2bec3' }}>No trouble comments with timestamps to display.</p>
  }

  const yCount = conceptNames.length
  const chartHeight = Math.max(250, yCount * 52 + 80)

  return (
    <ResponsiveContainer width="100%" height={chartHeight}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
        <XAxis
          type="number"
          dataKey="x"
          domain={[0, duration || 'auto']}
          tickFormatter={fmtTs}
          label={{ value: 'Video timestamp', position: 'insideBottom', offset: -20, fontSize: 12 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={[-0.5, yCount - 0.5]}
          ticks={Array.from({ length: yCount }, (_, i) => i)}
          tickFormatter={(v) => {
            const name = conceptNames[Math.round(v)] || ''
            return name.length > 24 ? name.slice(0, 24) + '…' : name
          }}
          width={175}
          tick={{ fontSize: 11 }}
        />
        <ZAxis range={[40, 40]} />
        <Tooltip content={<CustomTooltip />} />
        <Scatter data={data}>
          {data.map((entry, i) => (
            <Cell key={i} fill={LABEL_COLOR[entry.label] || '#95a5a6'} fillOpacity={0.85} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  )
}
