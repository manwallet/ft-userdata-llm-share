# 配置文件说明

## 重要提示

在使用本项目前，您需要配置以下敏感信息。**请勿将包含真实密钥的配置文件提交到版本控制系统！**

## 配置步骤

### 1. LLM API 配置

在 `user_data/config.json` 的 `llm_config` 部分填写：

```json
"llm_config": {
    "api_base": "http://localhost:3120",
    "api_key": "your-api-key-here",
    "model": "qwen/qwen3-coder-30b",
    "embedding_model": "text-embedding-bge-m3"
}
```

### 2. 交易所 API 配置

在 `user_data/config.json` 的 `exchange` 部分填写：

```json
"exchange": {
    "name": "binance",
    "key": "your-binance-api-key",
    "secret": "your-binance-secret",
    ...
}
```

**安全建议：**

- 使用子账户进行测试
- 限制 API 权限（仅开启交易权限，不开启提现权限）
- 建议先使用 `"dry_run": true` 模式进行模拟交易

### 3. 通知配置（可选）

#### Telegram 通知

```json
"telegram": {
    "enabled": true,
    "token": "your-telegram-bot-token",
    "chat_id": "your-telegram-chat-id"
}
```

#### Discord 通知

```json
"discord": {
    "enabled": true,
    "webhook_url": "your-discord-webhook-url",
    ...
}
```

### 4. API Server 配置

```json
"api_server": {
    "enabled": true,
    "jwt_secret_key": "your-jwt-secret-key",
    "username": "your-username",
    "password": "your-password"
}
```

生成安全的 JWT 密钥：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 环境变量方式（推荐）

您也可以使用环境变量来配置敏感信息：

1. 复制 `.env.example` 为 `.env`
2. 在 `.env` 中填写实际值
3. 修改代码以支持从环境变量读取

## 安全检查清单

在分享代码或提交到 GitHub 前，请确保：

- [ ] 所有 API 密钥已移除或替换为占位符
- [ ] 交易所密钥已清空
- [ ] JWT 密钥已清空
- [ ] Telegram/Discord 令牌已清空
- [ ] 日志文件不包含敏感信息
- [ ] 数据库文件未包含在版本控制中
- [ ] `.gitignore` 已正确配置
- [ ] `.env` 文件未被提交

## 测试配置

配置完成后，建议先使用模拟模式测试：

```bash
# 确保 config.json 中设置了：
"dry_run": true,
"dry_run_wallet": 1340,
```

这样可以在不使用真实资金的情况下测试策略。

## 获取帮助

如果您在配置过程中遇到问题，请查看：

- Freqtrade 官方文档：<https://www.freqtrade.io/>
- 本项目的 README.md 文件
