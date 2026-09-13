import type { ReactNode } from "react";

export type MathPart = { text: string; kind: "base" | "sub" | "sup" };

const SYMBOLS: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", Gamma: "Γ", delta: "δ", Delta: "Δ",
  epsilon: "ε", zeta: "ζ", eta: "η", theta: "θ", Theta: "Θ", iota: "ι",
  kappa: "κ", lambda: "λ", Lambda: "Λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π",
  Pi: "Π", rho: "ρ", sigma: "σ", Sigma: "Σ", tau: "τ", upsilon: "υ",
  phi: "φ", Phi: "Φ", chi: "χ", psi: "ψ", Psi: "Ψ", omega: "ω", Omega: "Ω",
  hbar: "ℏ", sum: "Σ", inf: "∞", infty: "∞",
};

const SYMBOL_RE = new RegExp(`\\b(${Object.keys(SYMBOLS).join("|")})\\b`, "g");

const SUB_BASES = new Set([...Object.keys(SYMBOLS), "dE", "dV", "dr"]);

function operators(s: string): string {
  return s
    .replace(/->/g, "→")
    .replace(/<=/g, "≤")
    .replace(/>=/g, "≥")
    .replace(/!=/g, "≠")
    .replace(/\+-/g, "±")
    .replace(/\bSchrodinger\b/g, "Schrödinger")
    .replace(
      /(\d)e([-+]?\d+)/g,
      (_all, d: string, exp: string) => `${d}×10^${exp.replace(/^\+/, "")}`,
    )
    .replace(/<([^<>]{1,40})>/g, (all, inner: string) =>
      inner.includes("|") || /^[A-Za-z0-9]{1,3}$/.test(inner) ? `⟨${inner}⟩` : all,
    );
}

function symbols(s: string, raised: boolean): string {
  const out = s.replace(SYMBOL_RE, (m) => SYMBOLS[m]);
  return raised ? out.replace(/-/g, "−") : out;
}

const SUP_RE = /^\^(?:\{([^}]*)\}|(-?\d+(?:\.\d+)?(?:\/\d+)?)|([A-Za-z]))/;
const SUB_RE = /^_(?:\{([^}]*)\}|([A-Za-z0-9]+(?:,[A-Za-z0-9-]+)*))/;

export function mathParts(src: string): MathPart[] {
  const s = operators(src);
  const parts: MathPart[] = [];
  let base = "";
  const flush = () => {
    if (base) parts.push({ text: symbols(base, false), kind: "base" });
    base = "";
  };
  for (let i = 0; i < s.length; ) {
    const rest = s.slice(i);
    const sup = rest.match(SUP_RE);
    if (sup) {
      const text = sup[1] ?? sup[2] ?? sup[3];
      flush();
      parts.push({ text: symbols(text, true), kind: "sup" });
      i += sup[0].length;
      continue;
    }
    const sub = rest.match(SUB_RE);
    if (sub && subscriptable(base) && !/^[_.]/.test(rest.slice(sub[0].length))) {
      const text = sub[1] ?? sub[2];
      flush();
      parts.push({ text: symbols(text, true), kind: "sub" });
      i += sub[0].length;
      continue;
    }
    if (/^\d(?![0-9A-Za-z])/.test(rest) && /(?:^|[^A-Za-z0-9_])[A-Za-z]$/.test(base)) {
      const digit = s[i];
      flush();
      parts.push({ text: digit, kind: "sub" });
      i += 1;
      continue;
    }
    base += s[i];
    i += 1;
  }
  flush();
  return parts;
}

function subscriptable(base: string): boolean {
  const word = base.match(/[A-Za-zΑ-Ωα-ω]+$/);
  if (!word) return /[⟩)\]|0-9]$/.test(base);
  return word[0].length === 1 || SUB_BASES.has(word[0]);
}

export function Notation({ children }: { children: string }) {
  return (
    <span className="math-inline">
      {mathParts(children).map((p, i) =>
        p.kind === "sup" ? (
          <sup key={i}>{p.text}</sup>
        ) : p.kind === "sub" ? (
          <sub key={i}>{p.text}</sub>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </span>
  );
}

const RISE = { sup: -3.5, sub: 2.5, base: 0 } as const;

export function mathTspans(src: string): ReactNode[] {
  let at = 0;
  return mathParts(src).map((p, i) => {
    const dy = RISE[p.kind] - at;
    at = RISE[p.kind];
    const className = p.kind === "base" ? undefined : "exponent";
    return (
      <tspan key={i} className={className} dy={dy}>
        {p.text}
      </tspan>
    );
  });
}
