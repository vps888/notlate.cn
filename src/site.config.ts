export const homeCategories = [
  {
    label: "AI 编译器",
    id: "ai-compiler",
    href: "/category/ai-compiler",
    description: "MLIR、ONNX-MLIR、IREE、TVM、Triton 与 Ascend 编译器实现分析。",
  },
  {
    label: "推荐系统与机器学习",
    id: "recommender-systems",
    href: "/category/recommender-systems",
    description: "推荐系统、Embedding、机器学习、计算广告与模型评估笔记。",
  },
  {
    label: "工程与算法",
    id: "engineering",
    href: "/category/engineering",
    description: "工程复盘、算法基础和长期积累的可检索技术资料。",
  },
] as const;

export const siteConfig = {
  name: "NotLate · AI 编译器笔记",
  shortName: "NotLate",
  domain: "notlate.cn",
  tagline: "MLIR、AI 编译器与机器学习工程实践",
  description: "NotLate 的技术博客，记录 MLIR、AI 编译器、推荐系统、机器学习与工程实践。",
  author: "折腾侠",
  siblingSite: {
    name: "折腾侠",
    description: "AI 工具、AI Coding、开发环境与网络折腾笔记",
    href: "https://zhetengxia.com/",
  },
  nav: [
    { label: "首页", href: "/" },
    ...homeCategories.map(({ label, href }) => ({ label, href })),
    { label: "折腾博客 ↗", href: "https://zhetengxia.com/", external: true },
  ],
} as const;
