from pathlib import Path
import tensorrt as trt


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0


def export_engine(file, half, verbose=False, prefix='TensorRT:'):
    try:
        onnx = file.with_suffix('.onnx')

        print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE
        trt.init_libnvinfer_plugins(logger, namespace='')
        # ctypes.cdll.LoadLibrary('LayerNorm/LayerNorm.so')
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        # config.max_workspace_size = workspace * 1 << 30

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        print(f'{prefix} Network Description:')
        for inp in inputs:
            print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
        profile = builder.create_optimization_profile()
        profile.set_shape('decoder_input_ids', [3, 1], [3, 20],
                                  [3, 50])
        profile.set_shape('encoder_hidden_states', [3, 256, 768], [3, 512,768],
                                  [3, 768, 768])
        profile.set_shape('encoder_attention_mask', [3, 256], [3, 512],
                                  [3, 768])
        config.add_optimization_profile(profile)
     
        half &= builder.platform_has_fast_fp16
        print(f'{prefix} building FP{16 if half else 32} engine in {f}')

        if half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_serialized_network(network, config) as engine, open(f, 'wb') as t:
            t.write(engine)
      
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


export_engine(Path('output/onnx/florence2_dec_model.onnx'), True,verbose=False)
