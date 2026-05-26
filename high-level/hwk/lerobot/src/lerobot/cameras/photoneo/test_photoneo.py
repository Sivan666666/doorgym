#!/usr/bin/env python3
"""
Photoneo 相机测试脚本
用途：测试 LeRobot 集成的 PhotoneoCamera 类
"""
import os
import sys
import time
import numpy as np
import argparse
from loguru import logger
from pathlib import Path
import signal
import sys

# 引入 LeRobot 路径
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

# 导入 LeRobot 相机类
from lerobot.cameras.photoneo.camera_photoneo import PhotoneoCamera
from lerobot.cameras.photoneo.configuration_photoneo import PhotoneoCameraConfig


# ✅ 全局变量，用于 Ctrl+C 处理
_global_camera = None

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    logger.warning("\n⚠️  收到中断信号 (Ctrl+C)，正在清理...")
    global _global_camera
    if _global_camera is not None and _global_camera.is_connected:
        logger.info("正在断开相机连接...")
        _global_camera.disconnect()
    logger.info("清理完成，退出程序")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

def test_photoneo_basic(config: PhotoneoCameraConfig):
    """基础测试：连接、同步读取、断开"""
    global _global_camera  # ✅ 使用全局变量
    
    logger.info("=" * 60)
    logger.info("测试 1: 基础功能 (同步读取)")
    logger.info("=" * 60)
    
    camera = PhotoneoCamera(config)
    _global_camera = camera  # ✅ 保存到全局变量
    
    try:
        # 1. 测试连接
        logger.info("\n[1/3] 测试连接...")
        camera.connect()
        assert camera.is_connected, "相机连接失败"
        logger.success("✅ 连接成功")
        
        # 2. 测试同步读取
        logger.info("\n[2/3] 测试同步读取 (3 次)...")
        for i in range(3):
            t_start = time.time()
            pcd = camera.read()
            t_elapsed = (time.time() - t_start) * 1000
            
            logger.info(f"  Frame {i+1}:")
            logger.info(f"    点云数量: {len(pcd)}")
            logger.info(f"    读取耗时: {t_elapsed:.1f} ms")
            logger.info(f"    帧率: {1000/t_elapsed:.1f} FPS")
            
            if len(pcd) > 0:
                logger.info(f"    点云范围: X=[{pcd[:, 0].min():.3f}, {pcd[:, 0].max():.3f}] m")
                logger.info(f"             Y=[{pcd[:, 1].min():.3f}, {pcd[:, 1].max():.3f}] m")
                logger.info(f"             Z=[{pcd[:, 2].min():.3f}, {pcd[:, 2].max():.3f}] m")
            
            time.sleep(0.2)
        
        logger.success("✅ 同步读取测试通过")
        
        # 3. 测试断开
        logger.info("\n[3/3] 测试断开...")
        camera.disconnect()
        
        # ✅ 添加短暂延迟，确保断开完成
        time.sleep(0.5)
        
        # 检查状态
        logger.info(f"  断开后状态: is_connected={camera.is_connected}")
        assert not camera.is_connected, "相机断开失败"
        logger.success("✅ 断开成功")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise
    finally:
        if camera.is_connected:
            logger.warning("⚠️  相机仍处于连接状态，强制断开...")
            camera.disconnect()
        _global_camera = None  # ✅ 清除全局变量


def test_photoneo_async(config: PhotoneoCameraConfig):
    """异步测试：后台线程采集"""
    global _global_camera
    
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: 异步读取")
    logger.info("=" * 60)
    
    camera = PhotoneoCamera(config)
    _global_camera = camera
    
    try:
        # 1. 连接
        logger.info("\n[1/4] 连接相机...")
        camera.connect()
        logger.success("✅ 连接成功")
        
        # 2. 测试第一次异步读取（会启动后台线程）
        logger.info("\n[2/4] 首次异步读取（启动后台线程）...")
        start_time = time.perf_counter()
        pcd = camera.async_read(timeout_ms=2000)
        first_read_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"  首次读取:")
        logger.info(f"    点云数量: {len(pcd)}")
        logger.info(f"    总耗时: {first_read_time:.1f} ms (包含线程启动)")
        if len(pcd) > 0:
            logger.info(f"    中心点: [{pcd.mean(axis=0)[0]:.3f}, {pcd.mean(axis=0)[1]:.3f}, {pcd.mean(axis=0)[2]:.3f}] m")
        
        # 3. 测试后续异步读取（从缓存获取）
        logger.info("\n[3/4] 连续异步读取（从缓存）...")
        logger.info("  说明: 显示的是「访问缓存」的耗时，不是采集耗时")
        
        cache_times = []
        for i in range(5):
            start_time = time.perf_counter()
            pcd = camera.async_read(timeout_ms=2000)
            cache_time = (time.perf_counter() - start_time) * 1000
            cache_times.append(cache_time)
            
            logger.info(f"  Frame {i+1}:")
            logger.info(f"    点云数量: {len(pcd)}")
            logger.info(f"    缓存访问耗时: {cache_time:.3f} ms")
            
            time.sleep(0.1)  # 短暂延迟
        
        avg_cache_time = np.mean(cache_times)
        logger.info(f"\n  平均缓存访问耗时: {avg_cache_time:.3f} ms")
        logger.success(f"  ✅ 异步读取的优势: 几乎零延迟访问最新数据")
        
        # 4. 测试后台采集性能
        logger.info("\n[4/4] 测量后台线程采集频率...")
        logger.info("  方法: 连续获取 10 帧，检测数据更新间隔")
        
        prev_pcd = camera.async_read(timeout_ms=2000)
        update_intervals = []
        
        for i in range(10):
            time.sleep(0.05)  # 50ms 轮询间隔
            start_time = time.perf_counter()
            
            # 持续轮询直到数据更新
            while True:
                curr_pcd = camera.async_read(timeout_ms=2000)
                
                # 检测是否是新的点云（通过数组地址判断）
                if curr_pcd is not prev_pcd:
                    update_time = (time.perf_counter() - start_time) * 1000
                    update_intervals.append(update_time)
                    
                    logger.info(f"  更新 {i+1}: 检测到新数据 (等待 {update_time:.1f} ms)")
                    prev_pcd = curr_pcd
                    break
                
                time.sleep(0.01)  # 10ms 轮询间隔
        
        avg_interval = np.mean(update_intervals)
        fps = 1000.0 / (avg_interval + 50)  # 加上轮询延迟
        
        logger.info(f"\n  统计结果:")
        logger.info(f"    平均更新间隔: {avg_interval:.1f} ms")
        logger.info(f"    理论采集帧率: ~{fps:.1f} FPS")
        logger.success("✅ 异步读取测试通过")
        
        # 5. 测试线程状态
        logger.info("\n[5/5] 检查后台线程状态...")
        assert camera.thread is not None, "后台线程未启动"
        assert camera.thread.is_alive(), "后台线程已停止"
        logger.success("✅ 后台线程运行正常")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise
    finally:
        if camera.is_connected:
            camera.disconnect()
        _global_camera = None


def test_photoneo_with_processing(config: PhotoneoCameraConfig, args):
    """完整测试：采集 + 点云处理"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3: 采集 + 点云处理流程")
    logger.info("=" * 60)
    
    camera = PhotoneoCamera(config)
    
    # 导入你的点云处理函数
    try:
        project_root_parent = Path(__file__).resolve().parents[5]
        sys.path.insert(0, str(project_root_parent))
        from common.pcd_utils import process_point_cloud
        logger.success("✅ 成功导入 process_point_cloud")
    except ImportError as e:
        logger.warning(f"⚠️  无法导入 process_point_cloud: {e}")
        logger.info("  跳过点云处理步骤")
        process_point_cloud = None
    
    try:
        # 1. 连接
        camera.connect()
        logger.success("✅ 相机已连接")
        
        # 2. 完整流程测试
        logger.info(f"\n开始采集 {args.num_frames} 帧点云...")
        
        for i in range(args.num_frames):
            logger.info(f"\n--- Frame {i+1}/{args.num_frames} ---")
            
            # 计时
            start_time = time.perf_counter()
            
            # 使用异步读取
            raw_pcd = camera.async_read(timeout_ms=2000)
            capture_time = (time.perf_counter() - start_time) * 1000
            
            # 处理点云
            if process_point_cloud is not None:
                process_start = time.perf_counter()
                processed_pcd = process_point_cloud(
                    raw_pcd,visualize=True
                )
                process_time = (time.perf_counter() - process_start) * 1000
                
                logger.info(f"  采集耗时: {capture_time:.1f} ms")
                logger.info(f"  处理耗时: {process_time:.1f} ms")
                logger.info(f"  原始点数: {len(raw_pcd)}")
                logger.info(f"  处理后点数: {len(processed_pcd)}")
                
                # 保存第一帧
                if i == 0 and args.save_path:
                    np.save(args.save_path, processed_pcd)
                    logger.success(f"  💾 已保存到: {args.save_path}")
                
                # 可视化第一帧
                if i == 0 and args.visualize:
                    try:
                        import open3d as o3d
                        pcd_o3d = o3d.geometry.PointCloud()
                        pcd_o3d.points = o3d.utility.Vector3dVector(processed_pcd)
                        o3d.visualization.draw_geometries([pcd_o3d])
                    except ImportError:
                        logger.warning("⚠️  Open3D 未安装，跳过可视化")
            else:
                logger.info(f"  采集耗时: {capture_time:.1f} ms")
                logger.info(f"  原始点数: {len(raw_pcd)}")
            
            time.sleep(args.delay)
        
        logger.success("\n✅ 完整流程测试通过!")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_photoneo_find_cameras():
    """测试设备检测"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 4: 设备检测")
    logger.info("=" * 60)
    
    cameras = PhotoneoCamera.find_cameras()
    
    if len(cameras) == 0:
        logger.warning("⚠️  未检测到 Photoneo 设备")
    else:
        logger.success(f"✅ 检测到 {len(cameras)} 个设备:")
        for i, cam in enumerate(cameras):
            logger.info(f"  [{i+1}] {cam['name']}")
            logger.info(f"      ID: {cam['id']}")
            logger.info(f"      SN: {cam['serial_number']}")


def main():
    parser = argparse.ArgumentParser(
        description="Photoneo 相机 LeRobot 集成测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式:
  --test basic       : 基础功能测试（连接、同步读取、断开）
  --test async       : 异步读取测试（后台线程）
  --test processing  : 完整流程测试（采集 + 点云处理）
  --test find        : 设备检测测试
  --test all         : 运行所有测试（默认）

示例:
  # 运行所有测试（使用默认外参）
  python test_photoneo.py
  
  # 只测试基础功能
  python test_photoneo.py --test basic
  
  # 自定义外参
  python test_photoneo.py --camera_pos 1.541 0.139 0.759 --camera_quat 0.585 0.606 -0.400 -0.362
  
  # 完整流程测试 + 可视化
  python test_photoneo.py --test processing --num_frames 5 --visualize --save_path test_pcd.npy
        """
    )
    
    # 测试模式
    parser.add_argument("--test", type=str, default="all",
                       choices=["basic", "async", "processing", "find", "all"],
                       help="测试模式")
    
    # 相机配置
    parser.add_argument("--photoneo_id", type=str, default="PAG-076",
                       help="Photoneo 设备 ID (不含前缀)")
    
    parser.add_argument("--calib_path", type=str, default=None,
                       help="外参标定文件路径 (txt 格式)")
    
    parser.add_argument("--camera_pos", type=float, nargs=3, 
                       default= [-0.03846401,-0.11231157,1.13300097],
                       metavar=('X', 'Y', 'Z'),
                       help="相机在世界坐标系的位置 (米)")
    
    parser.add_argument("--camera_quat", type=float, nargs=4,
                      default=[0.70629295,-0.69512124,0.09361616,-0.09587879],
                       metavar=('X', 'Y', 'Z', 'W'),
                       help="相机在世界坐标系的旋转四元数 [x, y, z, w]")
    
    # 测试参数
    parser.add_argument("--num_frames", type=int, default=3,
                       help="采集帧数（仅 processing 模式）")
    
    parser.add_argument("--delay", type=float, default=0.5,
                       help="帧间延迟 (秒)")
    
    parser.add_argument("--visualize", action="store_true",
                       help="使用 Open3D 可视化第一帧点云")
    
    parser.add_argument("--save_path", type=str, default=None,
                       help="保存第一帧点云的路径 (npy 格式)")
    
    # ✅ 修改默认等待时间为 5 秒（或更长）
    parser.add_argument("--wait_time", type=float, default=5.0,
                       help="测试间等待时间 (秒)，默认 5.0 秒（Photoneo 设备需要较长释放时间）")
    
    args = parser.parse_args()
    
    # 创建配置
    config = PhotoneoCameraConfig(
        device_id=args.photoneo_id,
        fps=25,
        translation=args.camera_pos,
        quaternion=args.camera_quat,
        calibration_path=args.calib_path,
    )
    
    logger.info("Photoneo 相机配置:")
    logger.info(f"  设备ID: {config.device_id}")
    logger.info(f"  外参平移: {config.translation}")
    logger.info(f"  外参四元数: {config.quaternion}")
    if config.calibration_path:
        logger.info(f"  标定文件: {config.calibration_path}")
    logger.info("")
    
    # 运行测试
    try:
        if args.test == "find" or args.test == "all":
            test_photoneo_find_cameras()
        
        if args.test == "basic" or args.test == "all":
            test_photoneo_basic(config)
            if args.test == "all":
                logger.info(f"\n⏳ 等待设备完全释放资源 ({args.wait_time}秒)...")
                time.sleep(args.wait_time)
        
        if args.test == "async" or args.test == "all":
            test_photoneo_async(config)
            if args.test == "all":
                logger.info(f"\n⏳ 等待设备完全释放资源 ({args.wait_time}秒)...")
                time.sleep(args.wait_time)
        
        if args.test == "processing" or args.test == "all":
            test_photoneo_with_processing(config, args)
        
        logger.success("\n" + "=" * 60)
        logger.success("✅ 所有测试通过!")
        logger.success("=" * 60)
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error(f"❌ 测试失败: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()