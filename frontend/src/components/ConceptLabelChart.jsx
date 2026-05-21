import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

const LABEL_MAP = {
  'expresses confusion or difficulty understanding the content': 'Confusion',
  'asks a clarifying question about the content':               'Clarifying',
  'expresses frustration with the explanation':                 'Frustration',
}

const COLORS = {
  Confusion:  '#e74c3c',
  Clarifying: '#f39c12',
  Frustration:'#c0392b',
}

export default function ConceptLabelChart({ conceptReports }) {
  const data = conceptReports
    .filter(r => r.matched_comment_count > 0)
    .map(r => {
      const row = {
        name: r.concept_name.length > 22 ? r.concept_name.slice(0, 22) + '…' : r.concept_name,
        full: r.concept_name,
        Confusion:   0,
        Clarifying:  0,
        Frustration: 0,
      }
      r.matched_comments?.forEach(mc => {
        const short = LABEL_MAP[mc.predicted_label]
        if (short) row[short]++
      })
      return row
    })

  if (data.length === 0) {
    return <p style={{ color: '#b2bec3' }}>No concepts with matched comments.</p>
  }

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 38 + 80)}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" allowDecimals={false} />
        <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 11 }} />
        <Tooltip labelFormatter={(_, payload) => payload?.[0]?.payload?.full || ''} />
        <Legend />
        {Object.entries(COLORS).map(([key, color]) => (
          <Bar key={key} dataKey={key} stackId="a" fill={color} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
