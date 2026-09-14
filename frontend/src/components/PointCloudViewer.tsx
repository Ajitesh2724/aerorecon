import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import {
  Maximize2,
  Minimize2,
  Camera,
  Grid,
  Eye,
  Sliders,
  Compass,
  Layers,
  Box,
  Palette,
  Ruler,
  X,
} from 'lucide-react';

interface CameraPose {
  image_id?: number;
  image_name?: string;
  position: [number, number, number];
  rotation_matrix?: number[][];
  quaternion?: [number, number, number, number];
}

interface PointCloudViewerProps {
  sparsePlyUrl?: string;
  densePlyUrl?: string;
  meshUrl?: string;
  posesUrl?: string;
  scaleFactor?: number;
  title?: string;
  className?: string;
}

type ViewMode = 'sparse' | 'dense' | 'mesh';
type ColorMode = 'elevation' | 'rgb' | 'cyan';

export default function PointCloudViewer({
  sparsePlyUrl,
  densePlyUrl,
  meshUrl,
  posesUrl,
  scaleFactor = 1.0,
  title = '3D Reconstruction Viewer',
  className = '',
}: PointCloudViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Active view mode
  const [viewMode, setViewMode] = useState<ViewMode>(
    densePlyUrl ? 'dense' : sparsePlyUrl ? 'sparse' : meshUrl ? 'mesh' : 'sparse'
  );
  const [colorMode, setColorMode] = useState<ColorMode>('elevation');

  const [pointCount, setPointCount] = useState(0);
  const [cameraCount, setCameraCount] = useState(0);
  const [faceCount, setFaceCount] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // View settings
  const [pointSize, setPointSize] = useState(3.0);
  const [showCameras, setShowCameras] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [wireframeMesh, setWireframeMesh] = useState(false);

  // Measurement Tool
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [measureUnit, setMeasureUnit] = useState<'m' | 'ft'>('m');
  const [measureResult, setMeasureResult] = useState<{
    dist3D: number;
    distHoriz: number;
    heightDiff: number;
  } | null>(null);

  const measurePointsRef = useRef<THREE.Vector3[]>([]);
  const isMeasuringRef = useRef(false);
  isMeasuringRef.current = isMeasuring;

  // Scene references
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  // Objects in scene
  const pointsRef = useRef<THREE.Points | null>(null);
  const meshRef = useRef<THREE.Group | null>(null);
  const trajectoryGroupRef = useRef<THREE.Group | null>(null);
  const camerasGroupRef = useRef<THREE.Group | null>(null);
  const gridHelperRef = useRef<THREE.GridHelper | null>(null);
  const measureGroupRef = useRef<THREE.Group | null>(null);

  // Initialize Three.js scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight || 480;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e17);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 2000);
    camera.position.set(0, -10, 15);
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
    const ambient = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambient);
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight1.position.set(20, -30, 40);
    scene.add(dirLight1);
    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.6);
    dirLight2.position.set(-20, 30, -20);
    scene.add(dirLight2);

    // Ground Grid
    const grid = new THREE.GridHelper(40, 40, 0x38bdf8, 0x1e293b);
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

    const mGroup = new THREE.Group();
    scene.add(mGroup);
    measureGroupRef.current = mGroup;

    // Canvas click listener for measurement
    const handleCanvasClick = (e: MouseEvent) => {
      if (!isMeasuringRef.current || !cameraRef.current || !rendererRef.current) return;

      const rect = rendererRef.current.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );

      const raycaster = new THREE.Raycaster();
      raycaster.params.Points = { threshold: 0.35 };
      raycaster.setFromCamera(mouse, cameraRef.current);

      let targetPoint: THREE.Vector3 | null = null;

      if (pointsRef.current) {
        const hits = raycaster.intersectObject(pointsRef.current, false);
        if (hits.length > 0) {
          targetPoint = hits[0].point;
        }
      }
      if (!targetPoint && meshRef.current) {
        const hits = raycaster.intersectObjects(meshRef.current.children, true);
        if (hits.length > 0) {
          targetPoint = hits[0].point;
        }
      }
      if (!targetPoint) {
        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
        targetPoint = new THREE.Vector3();
        raycaster.ray.intersectPlane(plane, targetPoint);
      }

      if (!targetPoint) return;

      const group = measureGroupRef.current;
      if (!group) return;

      if (measurePointsRef.current.length >= 2) {
        measurePointsRef.current = [];
        group.clear();
        setMeasureResult(null);
      }

      measurePointsRef.current.push(targetPoint.clone());

      // Add point sphere
      const sphereGeom = new THREE.SphereGeometry(0.12, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
      const sphere = new THREE.Mesh(sphereGeom, sphereMat);
      sphere.position.copy(targetPoint);
      group.add(sphere);

      if (measurePointsRef.current.length === 2) {
        const p1 = measurePointsRef.current[0];
        const p2 = measurePointsRef.current[1];

        // Add connecting line
        const lineGeom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0xf59e0b, linewidth: 2 });
        group.add(new THREE.Line(lineGeom, lineMat));

        const s = scaleFactor || 1.0;
        const d3 = p1.distanceTo(p2) * s;
        const dh = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) * s;
        const dz = Math.abs(p2.z - p1.z) * s;

        setMeasureResult({
          dist3D: d3,
          distHoriz: dh,
          heightDiff: dz,
        });
      }
    };

    renderer.domElement.addEventListener('click', handleCanvasClick);

    // Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 480;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('click', handleCanvasClick);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, [scaleFactor]);

  // Clear measurements
  const clearMeasurement = () => {
    measurePointsRef.current = [];
    if (measureGroupRef.current) {
      measureGroupRef.current.clear();
    }
    setMeasureResult(null);
  };

  const toggleMeasuring = () => {
    if (isMeasuring) {
      clearMeasurement();
      setIsMeasuring(false);
    } else {
      clearMeasurement();
      setIsMeasuring(true);
    }
  };

  // Determine current active geometry URL
  const activePlyUrl = viewMode === 'dense' ? (densePlyUrl || sparsePlyUrl) : sparsePlyUrl;

  // Load Point Cloud (Sparse or Dense)
  useEffect(() => {
    if (viewMode === 'mesh') return;
    if (!activePlyUrl || !sceneRef.current) return;

    setLoading(true);
    setError(null);

    const loader = new PLYLoader();
    loader.load(
      activePlyUrl,
      (geometry) => {
        const scene = sceneRef.current;
        if (!scene) return;

        if (pointsRef.current) {
          scene.remove(pointsRef.current);
          pointsRef.current.geometry.dispose();
          pointsRef.current = null;
        }
        if (meshRef.current) {
          scene.remove(meshRef.current);
          meshRef.current = null;
        }

        geometry.computeBoundingBox();
        const box = geometry.boundingBox || new THREE.Box3();
        const size = new THREE.Vector3();
        box.getSize(size);

        geometry.center();

        const count = geometry.attributes.position.count;
        setPointCount(count);

        // Compute elevation gradient colors
        const positions = geometry.attributes.position.array;
        const countPts = geometry.attributes.position.count;
        const elevColors = new Float32Array(countPts * 3);
        const zMin = -size.z / 2;
        const zRange = size.z || 1.0;

        const colLow = new THREE.Color(0x06b6d4);   // cyan
        const colMid = new THREE.Color(0x6366f1);   // indigo
        const colHigh = new THREE.Color(0xf59e0b);  // amber

        for (let i = 0; i < countPts; i++) {
          const z = positions[i * 3 + 2];
          const t = THREE.MathUtils.clamp((z - zMin) / zRange, 0, 1);
          const c = new THREE.Color();
          if (t < 0.5) {
            c.lerpColors(colLow, colMid, t * 2);
          } else {
            c.lerpColors(colMid, colHigh, (t - 0.5) * 2);
          }
          elevColors[i * 3] = c.r;
          elevColors[i * 3 + 1] = c.g;
          elevColors[i * 3 + 2] = c.b;
        }

        if (geometry.attributes.color && colorMode === 'rgb') {
          // keep original RGB
        } else if (colorMode === 'elevation' || !geometry.attributes.color) {
          geometry.setAttribute('color', new THREE.BufferAttribute(elevColors, 3));
        }

        const material = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: colorMode !== 'cyan',
          color: colorMode === 'cyan' ? 0x38bdf8 : 0xffffff,
          sizeAttenuation: true,
        });

        const points = new THREE.Points(geometry, material);
        scene.add(points);
        pointsRef.current = points;

        const maxDim = Math.max(size.x, size.y, size.z, 6);
        if (cameraRef.current && controlsRef.current) {
          cameraRef.current.position.set(maxDim * 0.9, -maxDim * 1.2, maxDim * 0.9);
          cameraRef.current.lookAt(0, 0, 0);
          controlsRef.current.target.set(0, 0, 0);
          controlsRef.current.update();
        }

        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('Error loading PLY:', err);
        setError('Failed to load point cloud file.');
        setLoading(false);
      }
    );
  }, [activePlyUrl, viewMode, colorMode]);

  // Load Surface Mesh
  useEffect(() => {
    if (viewMode !== 'mesh') return;
    if (!meshUrl || !sceneRef.current) return;

    setLoading(true);
    setError(null);

    const loader = new OBJLoader();
    loader.load(
      meshUrl,
      (obj) => {
        const scene = sceneRef.current;
        if (!scene) return;

        if (pointsRef.current) {
          scene.remove(pointsRef.current);
          pointsRef.current = null;
        }
        if (meshRef.current) {
          scene.remove(meshRef.current);
          meshRef.current = null;
        }

        let totalFaces = 0;
        obj.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const m = child as THREE.Mesh;
            m.geometry.center();
            m.geometry.computeVertexNormals();
            if (m.geometry.index) {
              totalFaces += m.geometry.index.count / 3;
            }

            m.material = new THREE.MeshStandardMaterial({
              color: 0x94a3b8,
              roughness: 0.4,
              metalness: 0.1,
              wireframe: wireframeMesh,
              side: THREE.DoubleSide,
            });
          }
        });

        setFaceCount(Math.round(totalFaces));
        scene.add(obj);
        meshRef.current = obj;

        if (cameraRef.current && controlsRef.current) {
          cameraRef.current.position.set(10, -15, 12);
          cameraRef.current.lookAt(0, 0, 0);
          controlsRef.current.target.set(0, 0, 0);
          controlsRef.current.update();
        }

        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('Error loading OBJ:', err);
        setError('Mesh file is not yet available.');
        setLoading(false);
      }
    );
  }, [viewMode, meshUrl, wireframeMesh]);

  // Load Camera Trajectory & Frustums
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

        trajGroup.clear();
        camsGroup.clear();

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
          trajGroup.add(new THREE.Line(lineGeom, lineMat));
        }

        const frustumGeom = new THREE.ConeGeometry(0.2, 0.4, 4);
        frustumGeom.rotateX(Math.PI / 2);

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
          camsGroup.add(camMesh);
        });
      })
      .catch(() => {});
  }, [posesUrl]);

  // Update point size
  useEffect(() => {
    if (pointsRef.current) {
      (pointsRef.current.material as THREE.PointsMaterial).size = pointSize;
    }
  }, [pointSize]);

  // Visibility toggles
  useEffect(() => {
    if (camerasGroupRef.current) camerasGroupRef.current.visible = showCameras;
  }, [showCameras]);

  useEffect(() => {
    if (trajectoryGroupRef.current) trajectoryGroupRef.current.visible = showTrajectory;
  }, [showTrajectory]);

  useEffect(() => {
    if (gridHelperRef.current) gridHelperRef.current.visible = showGrid;
  }, [showGrid]);

  const setPresetView = useCallback((type: 'top' | 'isometric' | 'side') => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    if (type === 'top') {
      camera.position.set(0, 0, 20);
      camera.up.set(0, 1, 0);
    } else if (type === 'isometric') {
      camera.position.set(12, -12, 12);
      camera.up.set(0, 0, 1);
    } else if (type === 'side') {
      camera.position.set(18, 0, 4);
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

  const convertUnit = (meters: number) => {
    if (measureUnit === 'ft') {
      return `${(meters * 3.28084).toFixed(2)} ft`;
    }
    return `${meters.toFixed(2)} m`;
  };

  return (
    <div
      className={`glass-card relative flex flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0a0e17] ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none' : 'h-[540px]'
      } ${className}`}
    >
      {/* Top HUD Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between border-b border-white/[0.08] bg-[#0a0e17]/85 px-4 py-2.5 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex h-2.5 w-2.5 items-center justify-center rounded-full bg-cyan-400">
            <div className="h-1.5 w-1.5 animate-ping rounded-full bg-cyan-400" />
          </div>
          <span className="text-sm font-semibold tracking-wide text-slate-200">{title}</span>

          {/* Mode Switcher Tabs */}
          <div className="flex items-center gap-1 rounded-lg bg-white/[0.05] p-0.5 ml-2">
            {sparsePlyUrl && (
              <button
                onClick={() => setViewMode('sparse')}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition ${
                  viewMode === 'sparse'
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Layers className="h-3 w-3" /> Sparse (SfM)
              </button>
            )}

            {densePlyUrl && (
              <button
                onClick={() => setViewMode('dense')}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition ${
                  viewMode === 'dense'
                    ? 'bg-gradient-to-r from-indigo-500 to-cyan-500 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Box className="h-3 w-3" /> Dense Cloud
              </button>
            )}

            {meshUrl && (
              <button
                onClick={() => setViewMode('mesh')}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition ${
                  viewMode === 'mesh'
                    ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Grid className="h-3 w-3" /> Surface Mesh
              </button>
            )}
          </div>
        </div>

        {/* View Controls & Action Buttons */}
        <div className="flex items-center gap-1.5">
          {/* Measurement Button */}
          <button
            onClick={toggleMeasuring}
            title={isMeasuring ? 'Exit Measurement Tool' : '3D Distance Measurement Tool'}
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
              isMeasuring
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                : 'text-slate-400 hover:bg-white/[0.08] hover:text-white'
            }`}
          >
            <Ruler className="h-3.5 w-3.5" />
            {isMeasuring ? 'Measuring…' : 'Measure'}
          </button>

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
            title="Viewer Options"
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

      {/* Measurement HUD Flyout */}
      {isMeasuring && (
        <div className="absolute top-12 left-4 z-20 w-72 rounded-xl border border-amber-500/30 bg-[#111827]/95 p-3.5 shadow-2xl backdrop-blur-xl">
          <div className="flex items-center justify-between border-b border-white/[0.08] pb-2 mb-2">
            <span className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-amber-400">
              <Ruler className="h-3.5 w-3.5" /> 3D Metric Measurement
            </span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setMeasureUnit(measureUnit === 'm' ? 'ft' : 'm')}
                className="rounded bg-white/[0.06] px-1.5 py-0.5 text-[10px] font-mono text-cyan-300 hover:bg-white/[0.12]"
              >
                {measureUnit.toUpperCase()}
              </button>
              <button
                onClick={toggleMeasuring}
                className="rounded p-0.5 text-slate-400 hover:text-white"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 mb-2">
            Click two points in the 3D scene to calculate real-world metric distance.
          </p>

          {measureResult ? (
            <div className="space-y-1.5 rounded-lg bg-black/40 p-2.5 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">3D Distance:</span>
                <span className="font-bold text-amber-400">{convertUnit(measureResult.dist3D)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Horizontal (Δxy):</span>
                <span className="text-cyan-300">{convertUnit(measureResult.distHoriz)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Height Diff (Δz):</span>
                <span className="text-emerald-300">{convertUnit(measureResult.heightDiff)}</span>
              </div>
            </div>
          ) : (
            <div className="rounded-lg bg-black/30 p-2 text-center text-xs text-slate-500 italic">
              Click 1st point to start…
            </div>
          )}

          <button
            onClick={clearMeasurement}
            className="mt-2.5 w-full rounded-md bg-white/[0.06] py-1 text-center text-[11px] text-slate-300 hover:bg-white/[0.1]"
          >
            Clear Measurement
          </button>
        </div>
      )}

      {/* Settings Flyout Drawer */}
      {showSettings && (
        <div className="absolute top-12 right-4 z-20 w-64 rounded-xl border border-white/[0.1] bg-[#111827]/95 p-3.5 shadow-2xl backdrop-blur-xl">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Viewer Settings
          </p>

          <div className="space-y-3 text-xs text-slate-300">
            {viewMode !== 'mesh' && (
              <>
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
                    <Palette className="h-3.5 w-3.5 text-cyan-400" /> Color Mode
                  </span>
                  <select
                    value={colorMode}
                    onChange={(e) => setColorMode(e.target.value as ColorMode)}
                    className="rounded bg-black/40 px-2 py-0.5 text-xs text-slate-300 border border-white/[0.1]"
                  >
                    <option value="elevation">Elevation Ramp</option>
                    <option value="rgb">True Color (RGB)</option>
                    <option value="cyan">Tactical Cyan</option>
                  </select>
                </div>
              </>
            )}

            {viewMode === 'mesh' && (
              <div className="flex items-center justify-between border-t border-white/[0.06] pt-2">
                <span>Wireframe Mode</span>
                <input
                  type="checkbox"
                  checked={wireframeMesh}
                  onChange={(e) => setWireframeMesh(e.target.checked)}
                  className="rounded accent-cyan-400"
                />
              </div>
            )}

            <div className="flex items-center justify-between border-t border-white/[0.06] pt-2">
              <span className="flex items-center gap-1.5">
                <Camera className="h-3.5 w-3.5 text-cyan-400" /> Camera Frustums
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
                <Eye className="h-3.5 w-3.5 text-cyan-400" /> Flight Trajectory
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
                <Grid className="h-3.5 w-3.5 text-cyan-400" /> Ground Plane
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
      <div
        ref={mountRef}
        className={`h-full w-full ${isMeasuring ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing'}`}
      />

      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0a0e17]/85 backdrop-blur-sm">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
          <p className="mt-3 text-sm font-medium text-slate-300">
            {viewMode === 'mesh' ? 'Triangulating 3D Mesh…' : 'Loading 3D Point Cloud…'}
          </p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-[#0a0e17]/90">
          <p className="text-sm font-medium text-amber-400">{error}</p>
        </div>
      )}

      {/* Bottom Status Bar */}
      <div className="absolute bottom-2 left-3 right-3 z-10 flex items-center justify-between rounded-lg bg-black/50 px-3 py-1.5 text-[11px] text-slate-400 backdrop-blur-sm pointer-events-none">
        <div className="flex items-center gap-3">
          <span>
            {isMeasuring
              ? 'Click two points in scene to measure distance'
              : 'Drag to rotate · Right-click/Shift+drag to pan · Scroll to zoom'}
          </span>
          {pointCount > 0 && viewMode !== 'mesh' && (
            <span className="font-mono text-cyan-300">{pointCount.toLocaleString()} points</span>
          )}
          {faceCount > 0 && viewMode === 'mesh' && (
            <span className="font-mono text-emerald-300">{faceCount.toLocaleString()} triangles</span>
          )}
          {cameraCount > 0 && (
            <span className="font-mono text-indigo-300">{cameraCount} camera views</span>
          )}
        </div>
        <div className="font-mono text-cyan-400">
          {viewMode === 'mesh' ? '3D SURFACE TIN MESH' : colorMode === 'elevation' ? 'Z-UP ELEVATION RAMP' : 'TRUE RGB'}
        </div>
      </div>
    </div>
  );
}
