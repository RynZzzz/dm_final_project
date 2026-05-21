import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

export default function ConceptMatchChart({ conceptReports }) {
  const data = [...conceptReports]
    .sort((a, b) => b.matched_comment_count - a.matched_comment_count)
    .map(r => ({
      name: r.concept_name.length > 22 ? r.concept_name.slice(0, 22) + '…' : r.concept_name,
      full: r.concept_name,
      count: r.matched_comment_count,
    }))

  if (data.length === 0) {
    return <p style={{ color: '#b2bec3' }}>No concepts extracted.</p>
  }

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 38 + 60)}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 5, right: 50, left: 10, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" allowDecimals={false} />
        <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 11 }} />
        <Tooltip
          formatter={(v) => [v, 'Trouble comments']}
          labelFormatter={(_, payload) => payload?.[0]?.payload?.full || ''}
        />
        <Bar dataKey="count" radius={[0, 4, 4, 0]} label={{ position: 'right', fontSize: 11 }}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.count > 0 ? '#e17055' : '#dfe6e9'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
