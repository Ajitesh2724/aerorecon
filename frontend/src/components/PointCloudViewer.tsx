import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import {
  Maximize2,
  Minimize2,
  Camera,
  Grid,
  Eye,
  Sliders,
  Compass,
} from 'lucide-react';

interface CameraPose {
  image_id?: number;
  image_name?: string;
  position: [number, number, number];
  rotation_matrix?: number[][];
  quaternion?: [number, number, number, number];
}

interface PointCloudViewerProps {
  plyUrl: string;
  posesUrl?: string;
  title?: string;
  className?: string;
}

export default function PointCloudViewer({
  plyUrl,
  posesUrl,
  title = 'Sparse Point Cloud & Trajectory',
  className = '',
}: PointCloudViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pointCount, setPointCount] = useState(0);
  const [cameraCount, setCameraCount] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // View settings
  const [pointSize, setPointSize] = useState(3.0);
  const [showCameras, setShowCameras] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  // Scene references
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const trajectoryGroupRef = useRef<THREE.Group | null>(null);
  const camerasGroupRef = useRef<THREE.Group | null>(null);
  const gridHelperRef = useRef<THREE.GridHelper | null>(null);

  // Initialize Three.js scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight || 450;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f19);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
    camera.position.set(0, -5, 8);
    camera.up.set(0, 0, 1); // Z-up for aerial mapping
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controlsRef.current = controls;

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(10, 20, 15);
    scene.add(dirLight);

    // Coordinate Grid (XY plane ground)
    const grid = new THREE.GridHelper(30, 30, 0x38bdf8, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    gridHelperRef.current = grid;

    // Groups
    const trajGroup = new THREE.Group();
    scene.add(trajGroup);
    trajectoryGroupRef.current = trajGroup;

    const camsGroup = new THREE.Group();
    scene.add(camsGroup);
    camerasGroupRef.current = camsGroup;

    // Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 450;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, []);

  // Load PLY point cloud
  useEffect(() => {
    if (!plyUrl || !sceneRef.current) return;
    setLoading(true);
    setError(null);

    const loader = new PLYLoader();
    loader.load(
      plyUrl,
      (geometry) => {
        const scene = sceneRef.current;
        if (!scene) return;

        // Remove old points if any
        if (pointsRef.current) {
          scene.remove(pointsRef.current);
          pointsRef.current.geometry.dispose();
        }

        geometry.computeBoundingBox();
        const box = geometry.boundingBox || new THREE.Box3();
        const center = new THREE.Vector3();
        box.getCenter(center);
        const size = new THREE.Vector3();
        box.getSize(size);

        // Center geometry
        geometry.center();

        const count = geometry.attributes.position.count;
        setPointCount(count);

        // Check if colors exist, otherwise generate elevation gradient
        if (!geometry.attributes.color) {
          const positions = geometry.attributes.position.array;
          const colors = new Float32Array(positions.length);
          const zMin = -size.z / 2;
          const zRange = size.z || 1.0;

          const colLow = new THREE.Color(0x06b6d4);  // cyan
          const colMid = new THREE.Color(0x6366f1);  // indigo
          const colHigh = new THREE.Color(0xf59e0b); // amber

          for (let i = 0; i < count; i++) {
            const z = positions[i * 3 + 2];
            const t = THREE.MathUtils.clamp((z - zMin) / zRange, 0, 1);
            const c = new THREE.Color();
            if (t < 0.5) {
              c.lerpColors(colLow, colMid, t * 2);
            } else {
              c.lerpColors(colMid, colHigh, (t - 0.5) * 2);
            }
            colors[i * 3] = c.r;
            colors[i * 3 + 1] = c.g;
            colors[i * 3 + 2] = c.b;
          }
          geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }

        const material = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true,
        });

        const points = new THREE.Points(geometry, material);
        scene.add(points);
        pointsRef.current = points;

        // Adjust camera to fit bounding box
        const maxDim = Math.max(size.x, size.y, size.z, 5);
        if (cameraRef.current && controlsRef.current) {
          cameraRef.current.position.set(maxDim * 0.8, -maxDim * 1.2, maxDim * 0.9);
          cameraRef.current.lookAt(0, 0, 0);
          controlsRef.current.target.set(0, 0, 0);
          controlsRef.current.update();
        }

        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('Error loading PLY:', err);
        setError('Failed to load point cloud. The file may still be generating.');
        setLoading(false);
      }
    );
  }, [plyUrl]);

  // Load camera poses and build trajectory + frustums
  useEffect(() => {
    if (!posesUrl || !sceneRef.current) return;

    fetch(posesUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((poses: CameraPose[]) => {
        if (!Array.isArray(poses) || poses.length === 0) return;
        setCameraCount(poses.length);

        const trajGroup = trajectoryGroupRef.current;
        const camsGroup = camerasGroupRef.current;
        if (!trajGroup || !camsGroup) return;

        // Clear existing
        trajGroup.clear();
        camsGroup.clear();

        // 1. Build trajectory polyline
        const curvePoints: THREE.Vector3[] = [];
        poses.forEach((p) => {
          if (p.position && p.position.length === 3) {
            curvePoints.push(new THREE.Vector3(p.position[0], p.position[1], p.position[2]));
          }
        });

        if (curvePoints.length > 1) {
          const lineGeom = new THREE.BufferGeometry().setFromPoints(curvePoints);
          const lineMat = new THREE.LineBasicMaterial({
            color: 0x38bdf8,
            linewidth: 2,
            transparent: true,
            opacity: 0.85,
          });
          const line = new THREE.Line(lineGeom, lineMat);
          trajGroup.add(line);
        }

        // 2. Build camera frustums
        const frustumGeom = new THREE.ConeGeometry(0.18, 0.35, 4);
        frustumGeom.rotateX(Math.PI / 2); // align forward

        poses.forEach((p, idx) => {
          if (!p.position || p.position.length !== 3) return;

          const camMesh = new THREE.Mesh(
            frustumGeom,
            new THREE.MeshBasicMaterial({
              color: idx === 0 ? 0x10b981 : idx === poses.length - 1 ? 0xef4444 : 0x38bdf8,
              wireframe: true,
            })
          );
          camMesh.position.set(p.position[0], p.position[1], p.position[2]);

          if (p.rotation_matrix && p.rotation_matrix.length === 3) {
            const m = new THREE.Matrix4();
            const R = p.rotation_matrix;
            m.set(
              R[0][0], R[0][1], R[0][2], 0,
              R[1][0], R[1][1], R[1][2], 0,
              R[2][0], R[2][1], R[2][2], 0,
              0, 0, 0, 1
            );
            camMesh.rotation.setFromRotationMatrix(m);
          }

          camMesh.userData = p;
          camsGroup.add(camMesh);
        });
      })
      .catch((err) => {
        console.warn('Could not load camera poses:', err);
      });
  }, [posesUrl]);

  // Update point size
  useEffect(() => {
    if (pointsRef.current) {
      (pointsRef.current.material as THREE.PointsMaterial).size = pointSize;
    }
  }, [pointSize]);

  // Toggle visibility
  useEffect(() => {
    if (camerasGroupRef.current) camerasGroupRef.current.visible = showCameras;
  }, [showCameras]);

  useEffect(() => {
    if (trajectoryGroupRef.current) trajectoryGroupRef.current.visible = showTrajectory;
  }, [showTrajectory]);

  useEffect(() => {
    if (gridHelperRef.current) gridHelperRef.current.visible = showGrid;
  }, [showGrid]);

  // Preset view angles
  const setPresetView = useCallback((type: 'top' | 'isometric' | 'side') => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    if (type === 'top') {
      camera.position.set(0, 0, 15);
      camera.up.set(0, 1, 0);
    } else if (type === 'isometric') {
      camera.position.set(10, -10, 10);
      camera.up.set(0, 0, 1);
    } else if (type === 'side') {
      camera.position.set(15, 0, 2);
      camera.up.set(0, 0, 1);
    }
    controls.target.set(0, 0, 0);
    controls.update();
  }, []);

  const toggleFullscreen = () => {
    if (!mountRef.current) return;
    if (!document.fullscreenElement) {
      mountRef.current.parentElement?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  return (
    <div
      className={`glass-card relative flex flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0b0f19] ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none' : 'h-[520px]'
      } ${className}`}
    >
      {/* Top HUD Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between border-b border-white/[0.08] bg-[#0b0f19]/80 px-4 py-2.5 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex h-2.5 w-2.5 items-center justify-center rounded-full bg-emerald-400">
            <div className="h-1.5 w-1.5 animate-ping rounded-full bg-emerald-400" />
          </div>
          <span className="text-sm font-semibold tracking-wide text-slate-200">{title}</span>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="rounded bg-white/[0.06] px-2 py-0.5 font-mono">
              {pointCount > 0 ? `${pointCount.toLocaleString()} pts` : '0 pts'}
            </span>
            {cameraCount > 0 && (
              <span className="rounded bg-cyan-500/10 px-2 py-0.5 font-mono text-cyan-300">
                {cameraCount} poses
              </span>
            )}
          </div>
        </div>

        {/* View presets & Actions */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setPresetView('isometric')}
            title="Isometric View"
            className="rounded-lg p-1.5 text-slate-400 transition hover:bg-white/[0.08] hover:text-white"
          >
            <Compass className="h-4 w-4" />
          </button>
          <button
            onClick={() => setPresetView('top')}
            title="Top-Down Ortho View"
            className="rounded-lg px-2 py-1 text-xs font-medium text-slate-400 transition hover:bg-white/[0.08] hover:text-white"
          >
            Top
          </button>
          <button
            onClick={() => setShowSettings(!showSettings)}
            title="Display Options"
            className={`rounded-lg p-1.5 transition ${
              showSettings ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:bg-white/[0.08] hover:text-white'
            }`}
          >
            <Sliders className="h-4 w-4" />
          </button>
          <button
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            className="rounded-lg p-1.5 text-slate-400 transition hover:bg-white/[0.08] hover:text-white"
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Settings Flyout Drawer */}
      {showSettings && (
        <div className="absolute top-12 right-4 z-20 w-64 rounded-xl border border-white/[0.1] bg-[#111827]/95 p-3.5 shadow-2xl backdrop-blur-xl">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Viewer Controls
          </p>

          <div className="space-y-3 text-xs text-slate-300">
            <div>
              <div className="flex justify-between py-0.5">
                <span>Point Size</span>
                <span className="font-mono text-cyan-400">{pointSize}px</span>
              </div>
              <input
                type="range"
                min="1"
                max="8"
                step="0.5"
                value={pointSize}
                onChange={(e) => setPointSize(parseFloat(e.target.value))}
                className="w-full accent-cyan-400"
              />
            </div>

            <div className="flex items-center justify-between border-t border-white/[0.06] pt-2">
              <span className="flex items-center gap-1.5">
                <Camera className="h-3.5 w-3.5 text-cyan-400" /> Cameras
              </span>
              <input
                type="checkbox"
                checked={showCameras}
                onChange={(e) => setShowCameras(e.target.checked)}
                className="rounded accent-cyan-400"
              />
            </div>

            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <Eye className="h-3.5 w-3.5 text-cyan-400" /> Drone Trajectory
              </span>
              <input
                type="checkbox"
                checked={showTrajectory}
                onChange={(e) => setShowTrajectory(e.target.checked)}
                className="rounded accent-cyan-400"
              />
            </div>

            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <Grid className="h-3.5 w-3.5 text-cyan-400" /> Ground Grid
              </span>
              <input
                type="checkbox"
                checked={showGrid}
                onChange={(e) => setShowGrid(e.target.checked)}
                className="rounded accent-cyan-400"
              />
            </div>
          </div>
        </div>
      )}

      {/* Main 3D Canvas Mount */}
      <div ref={mountRef} className="h-full w-full cursor-grab active:cursor-grabbing" />

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0b0f19]/80 backdrop-blur-sm">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
          <p className="mt-3 text-sm font-medium text-slate-300">Loading 3D Point Cloud…</p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-[#0b0f19]/90">
          <p className="text-sm font-medium text-amber-400">{error}</p>
          <p className="mt-1 text-xs text-slate-500">
            Ensure the reconstruction SfM stage has finished.
          </p>
        </div>
      )}

      {/* Bottom status bar */}
      <div className="absolute bottom-2 left-3 right-3 z-10 flex items-center justify-between rounded-lg bg-black/40 px-3 py-1.5 text-[11px] text-slate-400 backdrop-blur-sm pointer-events-none">
        <div>
          <span>Drag to rotate · Right-click/Shift+drag to pan · Scroll to zoom</span>
        </div>
        <div className="font-mono text-cyan-400">Z-UP | Elevation Colormap</div>
      </div>
    </div>
  );
}
