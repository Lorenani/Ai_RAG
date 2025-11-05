# 🚀 Streamlit Cloud 完整部署指南

## ✅ 部署前检查清单

### 1. 项目文件已准备 ✅
- ✅ `app.py` 存在
- ✅ `requirements.txt` 包含 streamlit
- ✅ 代码已推送到 GitHub: https://github.com/Lorenani/Ai_RAG

### 2. 需要检查的事项
- [ ] 确认 GitHub 仓库是 Public（公开）
- [ ] 准备 DashScope API 密钥（如果没有，可以先不配置，使用测试数据）

---

## 📋 详细部署步骤

### 第一步：访问 Streamlit Cloud

1. **打开浏览器**，访问：https://share.streamlit.io/
2. **点击 "Sign in"**，使用 GitHub 账号登录
3. 首次使用需要授权 Streamlit Cloud 访问你的 GitHub 仓库

### 第二步：创建新应用

1. 登录后，点击页面右上角的 **"New app"** 按钮
2. 或者直接访问：https://share.streamlit.io/

### 第三步：配置应用信息

在创建页面填写以下信息：

#### 基本信息
- **Repository（仓库）**: 
  - 在下拉菜单中选择：`Lorenani/Ai_RAG`
  - 如果看不到，点击 "Refresh" 刷新

- **Branch（分支）**: 
  - 选择：`main`

- **Main file path（主文件路径）**: 
  - 输入：`app.py`

- **App URL（应用URL）**: 
  - 可以自定义，例如：`rag-enterprise-qa`
  - 最终链接会是：`https://rag-enterprise-qa.streamlit.app`

### 第四步：配置环境变量（重要！）

1. **点击 "Advanced settings"** 展开高级设置
2. **点击 "Secrets"** 标签
3. **在文本框中输入**：
   ```toml
   DASHSCOPE_API_KEY=你的DashScope_API密钥
   ```
   
   **格式示例**：
   ```toml
   DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

4. **如果没有 API 密钥**：
   - 可以先留空，部署后使用测试数据
   - 或者访问：https://dashscope.console.aliyun.com/ 申请免费API密钥

### 第五步：部署应用

1. **点击页面底部的 "Deploy" 按钮**
2. **等待部署**（通常需要 2-5 分钟）
3. 部署过程中会显示日志，可以查看进度

### 第六步：获得公开链接

部署成功后：
- ✅ 页面会显示 "Your app is live!"
- ✅ 你会得到一个公开链接，例如：
  ```
  https://rag-enterprise-qa.streamlit.app
  ```
- ✅ 这个链接可以直接分享给 HR

---

## 🎯 部署后的操作

### 1. 测试应用
- 访问你的应用链接
- 测试基本功能是否正常
- 如果使用测试数据，确保数据文件已包含在仓库中

### 2. 更新简历
将以下信息添加到简历：

```
项目名称：企业知识库 RAG 智能问答系统

技术栈：Python, Streamlit, FAISS, LangChain, DashScope/OpenAI

项目链接：
• 🌐 在线演示：https://你的应用名.streamlit.app
• 💻 GitHub源码：https://github.com/Lorenani/Ai_RAG
```

### 3. 后续更新
- 代码更新后，Streamlit Cloud 会自动重新部署
- 或者手动点击 "Reboot app" 重启应用

---

## ⚠️ 常见问题解决

### 问题1：找不到仓库
**解决**：
- 确保 GitHub 仓库是 Public（公开）
- 点击 "Refresh" 刷新仓库列表
- 检查是否已授权 Streamlit Cloud 访问

### 问题2：部署失败
**检查**：
- `requirements.txt` 是否包含所有依赖
- `app.py` 文件路径是否正确
- 查看部署日志中的错误信息

### 问题3：应用无法运行
**检查**：
- 环境变量是否正确配置
- API 密钥是否有效
- 数据文件是否存在（如果使用本地数据）

### 问题4：数据文件太大
**解决**：
- 使用 Git LFS（Large File Storage）
- 或使用云端存储（如 AWS S3）
- 或仅使用测试数据集

---

## 📝 部署检查清单

部署前确认：
- [ ] GitHub 仓库是 Public
- [ ] `app.py` 文件存在
- [ ] `requirements.txt` 包含 streamlit
- [ ] 代码已推送到 GitHub
- [ ] 准备 API 密钥（可选）

部署后确认：
- [ ] 应用链接可以访问
- [ ] 页面正常加载
- [ ] 基本功能可以测试
- [ ] 链接已添加到简历

---

## 🎉 完成！

部署成功后，你的项目就可以通过链接访问了！

**下一步**：
1. 测试应用功能
2. 更新简历，添加项目链接
3. 准备项目演示说明

祝部署顺利！🚀

