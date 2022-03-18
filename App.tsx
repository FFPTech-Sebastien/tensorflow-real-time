import {Dimensions, Platform, StyleSheet, View} from 'react-native';
import {cameraWithTensors} from "@tensorflow/tfjs-react-native";
import {Camera} from "expo-camera";
import {useEffect, useRef, useState} from "react";
import Canvas, {CanvasRenderingContext2D} from 'react-native-canvas';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import "@tensorflow/tfjs-backend-wasm";
import Svg, {Circle, G, Rect} from "react-native-svg";

const {width, height} = Dimensions.get('window');

const TensorCamera = cameraWithTensors(Camera);

const inputTensorWidth = 152;
const inputTensorHeight = 200;

export default function App() {
    let textureDims: { width: number; height: number; };
    if (Platform.OS === 'ios') {
        textureDims = {
            height: 1920,
            width: 1080,
        };
    } else {
        textureDims = {
            height: 1200,
            width: 1600,
        };
    }
    const [faceDetection, setFaceDetection] = useState<blazeface.BlazeFaceModel>();
    const [faces, setFaces] = useState<blazeface.NormalizedFace[]>();
    const canvasRef = useRef<Canvas>();
    const contextRef = useRef<CanvasRenderingContext2D>();

    const handleImageTensorReady = async (images: IterableIterator<tf.Tensor3D>,) => {
        const loop = async () => {
            if (faceDetection) {
                const image = images.next().value;
                const detectedFaces = await faceDetection.estimateFaces(image, false);
                setFaces(detectedFaces);
                tf.dispose(image);
            }
            requestAnimationFrame(loop);
        }
        loop();
    }

    const renderFaces = () => {
        if (!faces) return null;
        contextRef.current?.clearRect(0, 0, width, height);
        const face = faces[0];
        if (face) {
            const topLeft = face.topLeft as number[];
            const bottomRight = face.bottomRight as number[];
            const landmarks = (face.landmarks as number[][]).map((l, lIndex) => {
                return <Circle
                    key={`landmark_${lIndex}`}
                    cx={l[0]}
                    cy={l[1]}
                    r='2'
                    strokeWidth='0'
                    fill='blue'
                />;
            });
            const flipHorizontal = Platform.OS === 'ios' ? 1 : -1;
            return (
                <Svg
                    height='100%' width='100%'
                    viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}
                    scaleX={flipHorizontal}
                >
                    <G key="facebox">
                        <Rect x={topLeft[0]} y={topLeft[1]} width={bottomRight[0] - topLeft[0]}
                              height={bottomRight[1] - topLeft[1]}
                              fill="white"
                              fillOpacity={0.2}
                        />
                        {landmarks}
                    </G>
                </Svg>
            );
        }
        // faces.map((face, fIndex) => {
        //     const topLeft = face.topLeft as number[];
        //     const bottomRight = face.bottomRight as number[];
        //     console.log(topLeft, bottomRight);
        //
        //     (face.landmarks as number[][]).map((landmark) => {
        //         contextRef.current?.beginPath();
        //         contextRef.current!.fillStyle = 'blue';
        //         contextRef.current?.arc(landmark[0], landmark[1], 2, 0, 2 * Math.PI);
        //         contextRef.current?.fill();
        //     });
        //
        //     contextRef.current?.strokeRect(topLeft[0], topLeft[1], bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]);
        // })
    }

    const handleCanvas = async (can: Canvas) => {
        if (can) {
            can.width = width;
            can.height = height;
            const ctx: CanvasRenderingContext2D = can.getContext("2d");
            ctx.strokeStyle = "red";
            ctx.lineWidth = 3;
            ctx.fillStyle = "red";
            contextRef.current = ctx;
            canvasRef.current = can;
        }
    }

    useEffect(() => {
        (async () => {
            await Camera.requestCameraPermissionsAsync();
            await tf.setBackend("wasm");
            await tf.ready();
            console.log("tf ready!");
            const model = await blazeface.load();
            console.log("model ready");
            setFaceDetection(model);
        })();
    }, []);

    return (
        <>
            {faceDetection && <>
                <TensorCamera
                    type={Camera.Constants.Type.front}
                    style={{width: '100%', height: '100%'}}
                    cameraTextureHeight={textureDims.height}
                    cameraTextureWidth={textureDims.width}
                    resizeDepth={3}
                    resizeHeight={inputTensorHeight}
                    resizeWidth={inputTensorWidth}
                    autorender
                    onReady={handleImageTensorReady}
                    useCustomShadersToResize={false}
                />
                <View style={{
                    position: 'absolute',
                    width: '100%',
                    height: '100%',
                    zIndex: 999
                }}>
                    {renderFaces()}
                </View>
            </>}


            {/*<Canvas ref={handleCanvas} style={{*/}
            {/*    position: 'absolute',*/}
            {/*    width: '100%',*/}
            {/*    height: '100%',*/}
            {/*    zIndex: 99,*/}
            {/*}}/>*/}
        </>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
});
