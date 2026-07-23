---
title: "无需服务器个性化域名重定向到其他网站"
description: "1. 准备工作 1. 申请个人域名 1. 免费方式：从 ClouDNS.net (https://cloudns.net)上创建账号申请即可，右上角可以选择中文。如果实在不会操作，可以参考： 教你免费注册一个ClouDNS永久域名(保姆级教程） (https://blog.csdn.net/q…"
slug: "custom-domain-redirect"
legacyId: 18706147
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18706147"
pubDate: 2025-02-09
category: "工程与算法"
tags: ["工程与算法"]
featured: false
---

## 1. 准备工作
### 1. 申请个人域名
1. 免费方式：从 [ClouDNS.net](https://cloudns.net)上创建账号申请即可，右上角可以选择中文。如果实在不会操作，可以参考：[教你免费注册一个ClouDNS永久域名(保姆级教程）](https://blog.csdn.net/qq_56204872/article/details/135296571) ，还可以**自行搜索**或问**大模型**或**留言**。
2. 付费方式：从 主流域名注册商注册，国外：[namesilo](https://www.namesilo.com/domain/search-domains) 比较便宜，国内自行搜索**腾讯**、**阿里**等。

### 2. 使用Cloudflare免费托管个人域名
可以参考：[详细图文手把手教你阿里云注册域名如何托管到CloudFlare DNS服务](https://blog.csdn.net/dreamingsleeping/article/details/139745997)
其他平台申请的域名都类似，可以自行搜索如何托管或者留言。

## 2. 操作
1. 登录Cloudflare，选中要重定向的域名，例如我的是notlate.cn，找到**规则**->**页面规则**，截图如下：
![](https://img2024.cnblogs.com/blog/3599704/202502/3599704-20250209145434945-1293737322.png)

2. 创建两条转发URL规则，状态码301，填好要转发的网址，点击部署即可：
![](https://img2024.cnblogs.com/blog/3599704/202502/3599704-20250209145759895-649709639.png)

3. 添加完两条规则（分别是**带**www和**不带**www）之后效果如下：
![](https://img2024.cnblogs.com/blog/3599704/202502/3599704-20250209145537896-368299767.png)

4. 稍等几分钟重试即可，如果当前浏览器访问失败，可以尝试换个电脑浏览器、手机浏览器、清理当前浏览器缓存尝试。

#### 遇到其他问题可以留言。
