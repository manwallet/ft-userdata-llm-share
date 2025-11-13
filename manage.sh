#!/bin/bash

# Freqtrade LLM Strategy 管理脚本
# 功能：版本检测、自动部署、日志查看、数据清理

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONTAINER_NAME="freqtrade-llm"
DOCKER_COMPOSE_FILE="docker-compose.yml"

# 策略数据路径
LOG_DIR="user_data/logs"
DECISION_LOG="$LOG_DIR/llm_decisions.jsonl"
TRADE_LOG="$LOG_DIR/trade_experience.jsonl"
REWARD_LOG="$LOG_DIR/reward_learning.json"
TRADE_DB_PREFIX="user_data/tradesv3.sqlite"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}=================================================${NC}"
    echo -e "${BLUE}  Freqtrade LLM Strategy 管理工具${NC}"
    echo -e "${BLUE}=================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

ensure_log_file() {
    local file_path="$1"
    local dir_path

    dir_path="$(dirname "$file_path")"
    mkdir -p "$dir_path"

    if [ ! -f "$file_path" ]; then
        touch "$file_path"
    fi
}

# 检查Docker是否运行
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker未运行，请先启动Docker"
        exit 1
    fi
}

# 检查新版本
check_new_version() {
    echo -e "\n${BLUE}[1/4]${NC} 检查Freqtrade版本..."
    local local_id=""
    local remote_id=""
    local local_created=""
    local remote_created=""
    local local_short=""
    local remote_short=""

    if docker image inspect freqtradeorg/freqtrade:stable > /dev/null 2>&1; then
        local_id=$(docker image inspect --format '{{.Id}}' freqtradeorg/freqtrade:stable 2>/dev/null | head -n 1)
        local_created=$(docker image inspect --format '{{.Created}}' freqtradeorg/freqtrade:stable 2>/dev/null | head -n 1)
        local_short=${local_id#sha256:}
        local_short=${local_short:0:12}
        echo "  当前镜像: ${local_short:-未知} (${local_created:-未知时间})"
    else
        print_warning "本地未发现 freqtrade:stable 镜像，将获取最新版本..."
    fi

    print_warning "检查远程最新版本..."
    if ! docker pull freqtradeorg/freqtrade:stable > /dev/null 2>&1; then
        print_error "无法获取远程镜像，请检查网络或凭证"
        return 2
    fi

    remote_id=$(docker image inspect --format '{{.Id}}' freqtradeorg/freqtrade:stable 2>/dev/null | head -n 1)
    remote_created=$(docker image inspect --format '{{.Created}}' freqtradeorg/freqtrade:stable 2>/dev/null | head -n 1)

    if [ -z "$remote_id" ]; then
        print_error "未能读取远程镜像信息"
        return 2
    fi

    remote_short=${remote_id#sha256:}
    remote_short=${remote_short:0:12}

    if [ -z "$local_id" ]; then
        print_success "已下载最新镜像: ${remote_short} (${remote_created})"
        return 0
    fi

    if [ "$local_id" != "$remote_id" ]; then
        print_warning "发现新版本！"
        echo "  本地镜像: ${local_short}"
        echo "  最新镜像: ${remote_short} (${remote_created})"
        return 0
    else
        print_success "已是最新版本: ${remote_short} (${remote_created})"
        return 1
    fi
}

# 构建镜像
build_image() {
    echo -e "\n${BLUE}[2/4]${NC} 构建自定义镜像..."

    docker-compose build --no-cache

    print_success "镜像构建完成"
}

# 启动服务
start_service() {
    echo -e "\n${BLUE}[3/4]${NC} 启动服务..."

    # 停止旧容器
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "停止旧容器..."
        docker-compose down
    fi

    # 启动新容器
    docker-compose up -d

    print_success "服务已启动"

    # 等待容器完全启动
    echo "等待容器初始化..."
    sleep 5
}

# 查看日志
view_logs() {
    echo -e "\n${BLUE}[4/4]${NC} 查看实时日志..."
    echo -e "${YELLOW}按 Ctrl+C 退出日志查看${NC}\n"
    sleep 2

    docker logs -f $CONTAINER_NAME
}

tail_host_log() {
    local file_path="$1"
    local label="$2"

    ensure_log_file "$file_path"

    echo -e "\n${BLUE}${label}${NC}"
    echo -e "${YELLOW}按 Ctrl+C 退出日志查看${NC}\n"
    tail -n 40 -f "$file_path"
}

view_decision_logs() {
    tail_host_log "$DECISION_LOG" "LLM 决策日志 (llm_decisions.jsonl)"
}

view_trade_logs() {
    tail_host_log "$TRADE_LOG" "交易经验日志 (trade_experience.jsonl)"
}

# 清理数据
clean_data() {
    echo -e "\n${RED}=================================================${NC}"
    echo -e "${RED}  数据清理工具${NC}"
    echo -e "${RED}=================================================${NC}"
    echo ""
    echo -e "${YELLOW}警告：此操作将删除以下数据：${NC}"
    echo "  1. LLM 决策日志 (${DECISION_LOG})"
    echo "  2. 交易经验日志 (${TRADE_LOG})"
    echo "  3. 奖励学习数据 (${REWARD_LOG})"
    echo "  4. Freqtrade 运行日志 (${LOG_DIR}/freqtrade.log*)"
    echo "  5. 交易数据库 (${TRADE_DB_PREFIX}*)"
    echo ""
    echo -e "${RED}此操作不可恢复！${NC}"
    echo ""
    read -p "确定要清理所有数据？(输入 'yes' 确认): " confirm

    if [ "$confirm" != "yes" ]; then
        print_warning "取消清理操作"
        return
    fi

    echo ""
    print_warning "开始清理数据..."

    # 停止容器
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "停止容器..."
        docker-compose down
    fi

    # 清理文件
    echo "清理 LLM 决策日志..."
    rm -f "$DECISION_LOG"

    echo "清理交易经验日志..."
    rm -f "$TRADE_LOG"

    echo "清理奖励学习数据..."
    rm -f "$REWARD_LOG"

    echo "清理交易数据库..."
    rm -f "${TRADE_DB_PREFIX}"*

    echo "清理 Freqtrade 运行日志..."
    rm -f "$LOG_DIR"/freqtrade.log*

    print_success "数据清理完成！"
    echo ""
    read -p "是否立即重新启动服务？(y/n): " restart

    if [ "$restart" == "y" ] || [ "$restart" == "Y" ]; then
        start_service
        view_logs
    fi
}

# 快速启动
quick_start() {
    echo -e "\n${BLUE}快速启动服务...${NC}\n"

    # 检查容器是否已运行
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_success "容器已在运行中"
        view_logs
        return
    fi

    # 检查容器是否存在但已停止
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "启动已存在的容器..."
        docker-compose start
        print_success "服务已启动"
        sleep 3
        view_logs
        return
    fi

    # 容器不存在，需要创建
    print_warning "容器不存在，首次启动..."
    docker-compose up -d
    print_success "服务已启动"
    sleep 5
    view_logs
}

# 快速重启
quick_restart() {
    echo -e "\n${BLUE}快速重启服务...${NC}\n"

    docker-compose restart

    print_success "服务已重启"
    sleep 2
    view_logs
}

# 完整部署流程
full_deploy() {
    print_header
    check_docker

    # 检查版本
    if check_new_version; then
        read -p "是否更新到最新版本？(y/n): " update
        if [ "$update" != "y" ] && [ "$update" != "Y" ]; then
            print_warning "跳过更新"
        fi
    fi

    build_image
    start_service
    view_logs
}

# 主菜单
show_menu() {
    print_header
    echo ""
    echo "请选择操作："
    echo ""
    echo "  1) 快速启动 (直接启动 + 查看日志) ⚡"
    echo "  2) 快速重启 (重启容器 + 查看日志)"
    echo "  3) 完整部署 (检查版本 + 构建 + 启动 + 查看日志)"
    echo "  4) 查看容器日志"
    echo "  5) 查看LLM决策日志"
    echo "  6) 查看交易经验日志"
    echo "  7) 清理策略日志和数据库"
    echo "  8) 检查Freqtrade镜像版本"
    echo "  9) 停止服务"
    echo "  0) 退出"
    echo ""
    read -p "请输入选项 [0-9]: " choice

    case $choice in
        1)
            quick_start
            ;;
        2)
            quick_restart
            ;;
        3)
            full_deploy
            ;;
        4)
            view_logs
            ;;
        5)
            view_decision_logs
            ;;
        6)
            view_trade_logs
            ;;
        7)
            clean_data
            ;;
        8)
            check_docker
            check_new_version
            echo ""
            read -p "按任意键返回菜单..." -n 1
            show_menu
            ;;
        9)
            echo -e "\n${BLUE}停止服务...${NC}"
            docker-compose down
            print_success "服务已停止"
            ;;
        0)
            echo -e "\n${GREEN}再见！${NC}"
            exit 0
            ;;
        *)
            print_error "无效选项"
            sleep 1
            show_menu
            ;;
    esac
}

# 主程序
main() {
    check_docker

    # 如果有命令行参数，直接执行对应操作
    if [ $# -gt 0 ]; then
        case $1 in
            start)
                quick_start
                ;;
            restart)
                quick_restart
                ;;
            deploy)
                full_deploy
                ;;
            logs)
                view_logs
                ;;
            decisions|decision-logs)
                view_decision_logs
                ;;
            trades|trade-logs)
                view_trade_logs
                ;;
            version)
                check_new_version
                ;;
            clean)
                clean_data
                ;;
            stop)
                docker-compose down
                print_success "服务已停止"
                ;;
            *)
                echo "用法: $0 [start|restart|deploy|logs|decisions|trades|version|clean|stop]"
                echo ""
                echo "命令说明："
                echo "  start     - 快速启动（推荐日常使用）⚡"
                echo "  restart   - 快速重启"
                echo "  deploy    - 完整部署（检查更新+构建）"
                echo "  logs      - 查看容器日志"
                echo "  decisions - 查看LLM决策日志"
                echo "  trades    - 查看交易经验日志"
                echo "  version   - 检查Freqtrade镜像版本"
                echo "  clean     - 清理日志和数据库"
                echo "  stop      - 停止服务"
                echo ""
                echo "或者不带参数运行以显示交互式菜单"
                exit 1
                ;;
        esac
    else
        # 显示交互式菜单
        show_menu
    fi
}

# 执行主程序
main "$@"
