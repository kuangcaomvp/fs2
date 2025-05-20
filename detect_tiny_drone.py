import cv2
import numpy as np

"""
    初始背景更新率（alpha）：0.01（保守更新）-0.05（快速适应）
    方差阈值（var_thresh）：根据云层运动幅度调整，典型值30-100
    运动检测阈值（threshold值25）：根据场景噪声水平调整
"""


class RobustBackgroundSubtractor:
    def __init__(self, alpha=0.01, var_thresh=30):
        self.bg_model = None
        self.alpha = alpha  # 背景更新率
        self.var_thresh = var_thresh  # 局部方差阈值

    def apply(self, frame):
        if self.bg_model is None:
            self.bg_model = np.float32(frame)  # 初始化背景模型为float32
            return np.zeros_like(frame, dtype=np.uint8)

        # 计算局部方差（使用滑动窗口方差计算）
        frame_32 = np.float32(frame)
        diff = cv2.absdiff(frame_32, self.bg_model)
        diff_sq = diff ** 2

        # 使用15x15框计算局部方差
        var_map = cv2.boxFilter(diff_sq, -1, (15, 15), normalize=False)

        # 生成更新掩码（注意数据类型转换）
        update_mask = (var_map < self.var_thresh).astype(np.uint8) * 255  # 转换为uint8

        # 选择性背景更新（使用正确类型的mask）
        mask = cv2.accumulateWeighted(frame_32, self.bg_model, self.alpha, mask=update_mask)

        # 生成运动检测结果
        _, motion_mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
        self.bg_model = np.float32(frame)
        return motion_mask.astype(np.uint8)


class ORB:
    def __init__(self, **kwargs):
        super().__init__()
        self.detector = self.create_detector(**kwargs)
        self.descriptor = self.create_descriptor()

    def create_detector(self, **kwargs):
        nfeatures = kwargs.get('nfeatures', 5000)
        scaleFactor = kwargs.get('scaleFactor', 1.2)
        nlevels = kwargs.get('nlevels', 8)
        edgeThreshold = kwargs.get('edgeThreshold', 31)
        firstLevel = kwargs.get('firstLevel', 0)
        WTA_K = kwargs.get('WTA_K', 2)
        scoreType = kwargs.get('scoreType', cv2.ORB_HARRIS_SCORE)
        patchSize = kwargs.get('patchSize', 31)
        fastThreshold = kwargs.get('fastThreshold', 20)

        params = dict(
            nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels,
            edgeThreshold=edgeThreshold, firstLevel=firstLevel, WTA_K=WTA_K,
            scoreType=scoreType, patchSize=patchSize, fastThreshold=fastThreshold,
        )
        detector = cv2.ORB_create(**params)
        return detector

    def create_descriptor(self):
        descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
        return descriptor

    def get_keypoint_and_descriptor(self, image):
        keypoints = self.detector.detect(image, None)
        keypoints, descriptors = self.descriptor.compute(image, keypoints)
        return keypoints, descriptors


class FastDroneDetector:
    def __init__(self):
        # 运动检测参数
        self.orb = ORB()
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # 增量式网格聚类参数
        self.grid_size = 5  # 网格像素尺寸
        # 可以认为是阀值用于过滤无效目标框
        self.min_grid_count = 2  # 有效网格最小点数
        self.trajectory_window = 8  # 时间窗口长度（帧数）

        # 运动轨迹存储
        self.frame_counter = 0
        self.prev_pts = None
        self.prev_gray = None
        # 刷新的间隔时间
        self.interval = 2
        # 新增合并参数
        self.merge_threshold = 4  # 相邻网格最大间距（单位：网格数）

        # 手动并查集状态
        self.parent = {}
        self.rank = {}

        self.robust_bg = RobustBackgroundSubtractor()

    def _find(self, grid):
        """路径压缩查找"""
        if self.parent[grid] != grid:
            self.parent[grid] = self._find(self.parent[grid])
        return self.parent[grid]

    def _union(self, grid1, grid2):
        """按秩合并"""
        root1 = self._find(grid1)
        root2 = self._find(grid2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1

    def _detect_and_track(self, gray):
        status = None

        # 第一次初始化
        if self.frame_counter == 0:
            self.prev_pts, _ = self.orb.get_keypoint_and_descriptor(gray)
            self.prev_pts = np.array([kp.pt for kp in self.prev_pts], dtype=np.float32)
            self.prev_gray = gray.copy()
            self.frame_counter += 1
            return []

        # 间隔时间内进行一次角点检测刷新
        if self.frame_counter % self.interval == 0 or self.prev_pts is None:
            self.prev_pts, _ = self.orb.get_keypoint_and_descriptor(self.prev_gray)
            self.prev_pts = np.array([kp.pt for kp in self.prev_pts], dtype=np.float32)

        # 光流跟踪
        if self.prev_pts is not None and len(self.prev_pts) != 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                                           self.prev_pts, None,
                                                           **self.lk_params)

        if status is None:
            self.prev_pts = None
            self.prev_gray = gray.copy()
            self.frame_counter += 1
            return []

        # 筛选有效点
        valid_pts = curr_pts[status.ravel() == 1]
        self.prev_pts = valid_pts.copy()
        self.prev_gray = gray.copy()
        self.frame_counter += 1

        return valid_pts

    def _find_adjacent_grids(self, active_grids):
        """网格合并"""
        # 初始化并查集
        self.parent.clear()
        self.rank.clear()
        for grid in active_grids:
            self.parent[grid] = grid
            self.rank[grid] = 0

        # 构建邻接关系
        grid_list = list(active_grids)
        for i in range(len(grid_list)):
            (x1, y1) = grid_list[i]
            for j in range(i + 1, len(grid_list)):
                (x2, y2) = grid_list[j]
                if abs(x1 - x2) <= self.merge_threshold and abs(y1 - y2) <= self.merge_threshold:
                    self._union(grid_list[i], grid_list[j])

        # 提取连通区域
        clusters = {}
        for grid in grid_list:
            root = self._find(grid)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(grid)

        return clusters

    def _merge_grids(self, clusters):
        """合并相邻网格生成区域"""
        merged_regions = []

        for _, grids in clusters.items():
            if len(grids) == 0:
                continue

            # 计算合并后的边界
            x_coords = [g[0] for g in grids]
            y_coords = [g[1] for g in grids]

            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)

            # 转换为像素坐标
            x1 = min_x * self.grid_size
            y1 = min_y * self.grid_size
            x2 = (max_x + 1) * self.grid_size
            y2 = (max_y + 1) * self.grid_size

            merged_regions.append((x1, y1, x2, y2))

        return merged_regions

    def _incremental_cluster(self, points):
        """增量聚类"""
        active_grids = set()

        # 更新网格计数器
        for (x, y) in points:
            grid_x = int(x // self.grid_size)
            grid_y = int(y // self.grid_size)
            key = (grid_x, grid_y)
            active_grids.add(key)
        # 合并相邻网格
        clusters = self._find_adjacent_grids(active_grids)
        return self._merge_grids(clusters)

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # bg = self.robust_bg.apply(gray)
            # cv2.imshow('1', bg)

            """特征检测与跟踪"""
            # 特征跟踪
            tracked_pts = self._detect_and_track(gray)

            # 增量聚类
            targets = self._incremental_cluster(tracked_pts)

            # 可视化
            self._visualize(frame, targets)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _visualize(self, frame, targets):
        """高效可视化"""
        for bbox in targets:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.imshow('Fast Drone Detection', frame)


if __name__ == "__main__":
    detector = FastDroneDetector()
    detector.run(r'F:\uav\hw\11.mp4')  # F:\uav\hw\11.mp4  F:\1\4.mp4
