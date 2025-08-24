import React from "react";

type TickerItem = {
  rank?: number;                 // 1..10（サーバ付与）
  symbol: string;
  name?: string;
  theme?: string;
  score_pts?: number;            // 0..1000
  final_score_0_1?: number;      // 0..1
  vol_anomaly_score?: number;    // 0..1
  chart_url?: string;            // /charts/<DATE>/<SYMBOL>.png
};

export default function TickerCard({ item }: { item: TickerItem }) {
  const pts = typeof item.score_pts === "number"
    ? item.score_pts
    : Math.round(((item.final_score_0_1 ?? item.vol_anomaly_score ?? 0) * 1000));

  const rank = item.rank ?? null;
  const tier =
    rank === 1 ? "bg-amber-500 text-white"
    : rank === 2 ? "bg-gray-800 text-white"
    : rank === 3 ? "bg-orange-400 text-white"
    : "bg-gray-100 text-gray-700";

  return (
    <div className="rounded-2xl border p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {rank && (
            <span className={`inline-flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold ${tier}`}>
              {rank}
            </span>
          )}
          <div>
            <div className="text-xl font-semibold">{item.symbol}</div>
            {item.name && <div className="text-sm text-gray-500">{item.name}</div>}
          </div>
        </div>

        <div className="text-right">
          <div className="text-4xl font-extrabold leading-none tracking-tight">{pts}</div>
          <div className="text-xs text-gray-500">/ 1000</div>
        </div>
      </div>

      <div className="mt-4">
        <Tabs
          overview={<Overview item={item} pts={pts} />}
          technical={<TechnicalSnapshot item={item} />}
        />
      </div>
    </div>
  );
}

function Overview({ item, pts }: { item: TickerItem; pts: number }) {
  const vol = item.vol_anomaly_score ?? item.final_score_0_1 ?? 0;
  return (
    <div className="space-y-2">
      <p className="text-sm text-gray-600">
        現在のスコアは「異常出来高」のみで算出しています（将来拡張予定）。
      </p>
      <div className="flex flex-wrap gap-2 text-sm">
        <Badge label="Score" value={`${pts}/1000`} strong />
        <Badge label="Volume anomaly" value={vol.toFixed(2)} />
      </div>
    </div>
  );
}

function TechnicalSnapshot({ item }: { item: TickerItem }) {
  const src = item.chart_url ?? "";
  return (
    <div>
      {src ? (
        <img src={src} alt={`${item.symbol} weekly 3M`} className="w-full rounded-lg border" />
      ) : (
        <div className="text-sm text-gray-500">チャートは準備中です。</div>
      )}
    </div>
  );
}

function Badge({ label, value, strong = false }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className={`rounded-full border px-3 py-1 ${strong ? "border-black font-semibold" : ""}`}>
      <span className="mr-2 text-gray-500">{label}:</span>
      <span>{value}</span>
    </div>
  );
}

function Tabs({ overview, technical }: { overview: React.ReactNode; technical: React.ReactNode }) {
  const [tab, setTab] = React.useState<"overview" | "technical">("overview");
  const btn = (active: boolean) =>
    `rounded-full px-3 py-1 text-sm ${active ? "bg-black text-white" : "border"}`;
  return (
    <div>
      <div className="mb-3 flex gap-2">
        <button className={btn(tab === "overview")} onClick={() => setTab("overview")}>
          Overview
        </button>
        <button className={btn(tab === "technical")} onClick={() => setTab("technical")}>
          Technical snapshot
        </button>
      </div>
      <div>{tab === "overview" ? overview : technical}</div>
    </div>
  );
}
