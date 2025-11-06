#!/bin/bash

# Freqtrade LLM Strategy 管理脚本
# 功能：版本检测、自动部署、日志查看、数据清理

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONTAINER_NAME="freqtrade-llm"
DOCKER_COMPOSE_FILE="docker-compose.yml"

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

    # 获取本地镜像版本
    LOCAL_VERSION=$(docker images freqtradeorg/freqtrade:stable --format "{{.Tag}}" | head -n 1)

    # 拉取最新版本信息（不下载）
    print_warning "检查远程最新版本..."
    docker pull freqtradeorg/freqtrade:stable > /dev/null 2>&1 || true

    REMOTE_VERSION=$(docker images freqtradeorg/freqtrade:stable --format "{{.Tag}}" | head -n 1)

    if [ "$LOCAL_VERSION" != "$REMOTE_VERSION" ]; then
        print_warning "发现新版本！"
        echo "  本地版本: $LOCAL_VERSION"
        echo "  远程版本: $REMOTE_VERSION"
        return 0
    else
        print_success "已是最新版本: $LOCAL_VERSION"
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

# 清理数据
clean_data() {
    echo -e "\n${RED}=================================================${NC}"
    echo -e "${RED}  数据清理工具${NC}"
    echo -e "${RED}=================================================${NC}"
    echo ""
    echo -e "${YELLOW}警告：此操作将删除以下数据：${NC}"
    echo "  1. ChromaDB向量数据库 (RAG历史)"
    echo "  2. 交易数据库 (tradesv3.sqlite)"
    echo "  3. Freqtrade日志"
    echo "  4. LLM决策日志"
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
    echo "清理向量数据库..."
    rm -rf user_data/data/vector_store/*

    echo "清理交易数据库..."
    rm -f user_data/tradesv3.sqlite*

    echo "清理日志文件..."
    rm -f user_data/logs/freqtrade.log*
    rm -f user_data/logs/llm_decisions.jsonl
    rm -f user_data/logs/trade_experience.jsonl

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
    echo "  4) 只查看日志"
    echo "  5) 清理所有数据"
    echo "  6) 检查版本"
    echo "  7) 停止服务"
    echo "  0) 退出"
    echo ""
    read -p "请输入选项 [0-7]: " choice

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
            clean_data
            ;;
        6)
            check_docker
            check_new_version
            echo ""
            read -p "按任意键返回菜单..." -n 1
            show_menu
            ;;
        7)
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
            clean)
                clean_data
                ;;
            stop)
                docker-compose down
                print_success "服务已停止"
                ;;
            *)
                echo "用法: $0 [start|restart|deploy|logs|clean|stop]"
                echo ""
                echo "命令说明："
                echo "  start   - 快速启动（推荐日常使用）⚡"
                echo "  restart - 快速重启"
                echo "  deploy  - 完整部署（检查更新+构建）"
                echo "  logs    - 查看日志"
                echo "  clean   - 清理所有数据"
                echo "  stop    - 停止服务"
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
