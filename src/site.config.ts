export const homeCategories = [
  {
    label: "AI 编译器",
    id: "ai-compiler",
    href: "/category/ai-compiler",
    description: "MLIR、ONNX-MLIR、IREE、TVM、Triton 与 Ascend 编译器实现分析。",
  },
  {
    label: "推荐系统",
    id: "recommender-systems",
    href: "/category/recommender-systems",
    description: "推荐系统、Embedding、小红书推荐、机器学习与模型评估笔记。",
  },
  {
    label: "计算广告",
    id: "computational-advertising",
    href: "/category/computational-advertising",
    description: "ADX、DSP、SSP、竞价、CTR 与程序化交易技术笔记。",
  },
] as const;

export const siteConfig = {
  // These values are taken from the original cnblogs export (blog_Config).
  name: "AI技术小屋",
  shortName: "AI技术小屋",
  domain: "notlate.cn",
  subtitle: "生命不息，AI不止",
  description: "MLIR、AI 编译器、推荐系统、计算广告与折腾实践文章。",
  author: "折腾侠",
  authorDescription: "生命在于折腾",
  avatar: "https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/20220710_131340286_iOS.jpg",
  siblingSite: {
    name: "瞎折腾小屋",
    description: "AI 工具、AI Coding、开发环境与网络折腾笔记",
    href: "https://zhetengxia.com/",
  },
  externalLinks: [
    {
      name: "AI Systems Performance Engineering",
      href: "https://aispe.notlate.cn/",
    },
  ],
  nav: [
    { label: "首页", href: "https://notlate.cn/" },
    ...homeCategories.map(({ label, href }) => ({ label, href })),
    { label: "折腾实践 ↗", href: "https://zhetengxia.com/", external: true },
  ],
} as const;
