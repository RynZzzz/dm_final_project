import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Cell, ResponsiveContainer,
} from 'recharts'

const LABEL_SHORT = {
  'expresses confusion or difficulty understanding the content': 'Confusion',
  'asks a clarifying question about the content':               'Clarifying question',
  'expresses frustration with the explanation':                 'Frustration',
  'gives positive feedback or appreciation':                    'Positive feedback',
  'is neutral or off-topic':                                    'Neutral / off-topic',
}

function barColor(label) {
  if (label === 'Confusion' || label === 'Frustration') return '#e74c3c'
  if (label === 'Clarifying question') return '#f39c12'
  if (label === 'Positive feedback')   return '#2ecc71'
  return '#95a5a6'
}

export default function ClassificationChart({ labelDist }) {
  const data = Object.entries(labelDist).map(([k, v]) => ({
    label: LABEL_SHORT[k] || k,
    count: v,
  }))

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 5, right: 50, left: 10, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" allowDecimals={false} />
        <YAxis
          type="category"
          dataKey="label"
          width={150}
          tick={{ fontSize: 12 }}
        />
        <Tooltip formatter={(v) => [`${v} comments`, 'Count']} />
        <Bar dataKey="count" radius={[0, 4, 4, 0]} label={{ position: 'right', fontSize: 12 }}>
          {data.map((entry, i) => (
            <Cell key={i} fill={barColor(entry.label)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
