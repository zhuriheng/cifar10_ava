#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sys
import traceback
import signal
import mxnet as mx

from ava.params import params
from ava.log.logger import logger
from ava.monitor import mxnet as mxnet_monitor
from ava.train import base as train
from ava.utils.model import mxnet as model_utils
from ava.utils import utils

FIT_KWARGS_KEYS = ("eval_metric", "epoch_end_callback", "batch_end_callback",
                   "kvstore", "optimizer", "optimizer_params", "num_epoch",
                   "monitor")
CROP_CHANNELS = 3


class MXNetTrainingWorker(object):

    def __init__(self):
        self.train_ins = None
        self.train_config = {}
        self.solver_config = {}
        self.train_data = None
        self.val_data = None
        self.mod = None

    def prepare_train_config(self):
        """配置训练参数"""

        # AVA-SDK 获取训练参数
        """
        1) 获取所有配置 example
            param_dict = params.get_all()
            value1 = param_dict["key1"]
        2) 获取某项value
            value1 = params.get_value("key1", default=1)
        """
        snapshot_interval_epochs = params.get_value(
            "snapshot_interval_epochs", default=1)
        max_epochs = params.get_value("max_epochs", default=3)
        rand_crop = params.get_value(
            "rand_crop", default=True)
        rand_mirror = params.get_value(
            "rand_mirror", default=True)
        batch_size, actual_batch_size, val_batch_size = utils.get_batch_size()

        batch_size = params.get_value("batchSize", default=8)
        val_batch_size = params.get_value("valBatchSize", default=batch_size)
        use_gpu, cores = utils.get_cores()
        logger.info("Cores GPU=%s, count=%d", use_gpu, cores)
        actual_batch_size = batch_size if not use_gpu else batch_size * cores
        if use_gpu:
            val_batch_size *= cores

        # USING the trainning batch size as valadition batch size
        val_batch_size = actual_batch_size
        # crop_w, crop_h = utils.get_crop_size()
        # 根据模型的输入要求选择 crop_size
        crop_w = params.get_value("crop_w")
        crop_h = params.get_value("crop_h")

        # 添加监控
        snapshot_prefix = self.train_ins.get_snapshot_base_path() + "/snapshot"
        kv_store = "device"
        kv = mx.kvstore.create(kv_store)
        '''
        rank = int(kv.rank)
        if rank > 0:
            snapshot_prefix += "-%s" % rank
        '''

        batch_freq = 10     # 打印/上报指标的 batch 粒度
        batch_of_epoch = utils.ceil_by_level(
            float(utils.get_sampleset_num() / actual_batch_size))

        # AVA-SDK mxnet monitor callback 初始化
        batch_end_cb = self.train_ins.get_monitor_callback(
            "mxnet",
            batch_size=actual_batch_size,
            batch_freq=batch_freq)
        epoch_end_cb = [
            # mxnet default epoch callback
            mx.callback.do_checkpoint(
                snapshot_prefix, snapshot_interval_epochs),
            self.train_ins.get_epoch_end_callback(
                "mxnet", batch_of_epoch=batch_of_epoch, epoch_interval=snapshot_interval_epochs, other_files=[])
        ]

        # 训练参数，用户可以自行配置
        self.train_config = {
            "input_data_shape": (CROP_CHANNELS, crop_h, crop_w),
            "rand_crop": rand_crop,
            "rand_mirror": rand_mirror,
            "batch_size": batch_size,
            "actual_batch_size": actual_batch_size,
            "val_batch_size": val_batch_size,
            # fit_args
            "eval_metric": mxnet_monitor.full_mxnet_metrics(),  # AVA-SDK 获取mxnet metric 列表
            "epoch_end_callback": epoch_end_cb,
            "batch_end_callback": batch_end_cb,
            "kvstore": kv,
            "num_epoch": max_epochs,
        }

    def prepare_solver_config(self):
        use_gpu, cores = utils.get_cores()
        gpu_counts = cores if use_gpu else 0
        batch_size, actual_batch_size, val_batch_size = utils.get_batch_size()

        optimizer_params = {
            "momentum": params.get_value("momentum", default=0.9),
            "wd": params.get_value("wd", default=0.0005),
            "learning_rate": params.get_value("learning_rate", default=0.01),
            "lr_scheduler": mx.lr_scheduler.FactorScheduler(2000, factor=0.0001),
        }
        self.solver_config = {
            "gpu_counts": gpu_counts,
            # fit_args
            "optimizer": "SGD",
            "optimizer_params": optimizer_params,
        }

    def prepare_sampleset_data(self):
        """load sampleset data
        """

        # ava sdk 提供默认的数据集路径，如果用户需要读取其他地方的数据集，可自行配置路径
        train_data_path = self.train_ins.get_trainset_base_path() + "/cache/data.rec"
        self.train_data = mx.io.ImageRecordIter(
            path_imgrec=train_data_path,
            batch_size=self.train_config.get("actual_batch_size"),
            data_shape=self.train_config.get("input_data_shape"),
            shuffle=True,
            rand_crop=self.train_config.get("rand_crop"),
            rand_mirror=self.train_config.get("rand_mirror"))

        val_data_path = self.train_ins.get_valset_base_path() + "/cache/data.rec"
        if os.path.exists(val_data_path):
            self.val_data = mx.io.ImageRecordIter(
                path_imgrec=val_data_path,
                batch_size=self.train_config.get("val_batch_size"),
                data_shape=self.train_config.get("input_data_shape"))
        else:
            self.val_data = None

    def prepare_model(self):
        """load 网络模型，（更新输出层）
        """
        # 需要更新模型输出层种类数目的场景，用户自行决定模型文件的路径，非必要场景
        # 用户可以直接使用训练框架来读取模型
        # 替换成自己需要的模型名以及参数名 Riheng 2018/02/23
        origin_model_path = self.train_ins.get_model_base_path() + "/resnet-50-symbol.json"
        weight_file_path = self.train_ins.get_model_base_path() + "/resnet-50-0000.params"
        fixed_model_path = self.train_ins.get_model_base_path() + "/fixed_resnet-50-symbol.json"

        # AVA-SDK 获取数据集类型数 && 更新网络模型输出层
        output_layer_num = utils.get_sampleset_class_num()
        old_output_layer_name = model_utils.update_model_output_num(
            origin_model_path, fixed_model_path, output_layer_num)

        sym = mx.symbol.load(fixed_model_path)
        gpu_count = self.solver_config.get('gpu_counts', 0)
        ctx = [mx.cpu()] if gpu_count == 0 else [
            mx.gpu(i) for i in xrange(gpu_count)
        ]
        mod = mx.mod.Module(symbol=sym, context=ctx)

        mod.bind(data_shapes=self.train_data.provide_data,
                 label_shapes=self.train_data.provide_label)

        # 默认权值初始化方式
        mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian',
                                                   factor_type="in",
                                                   magnitude=2))
        # AVA-SDK 在替换网络输出层的场景下读取权重参数
        arg_params, aux_params = model_utils.load_model_params(
            weight_file_path, old_output_layer_name)
        if arg_params:
            logger.info("set pretrained weights")
            mod.set_params(arg_params, aux_params, allow_missing=True)

        self.mod = mod

    def get_fine_tune_model(self, symbol, arg_params, num_hidden, layer_name='flatten0'):
        """
        symbol: the pretrained network symbol
        arg_params: the argument parameters of the pretrained model
        num_hidden: tNumber of hidden nodes of the output.
        layer_name: the layer name before the last fully-connected layer
        """
        all_layers = symbol.get_internals()
        net = all_layers[layer_name+'_output']
        net = mx.symbol.FullyConnected(data=net, num_hidden=num_hidden, name='fc1')
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
        new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
        return (net, new_args)

    def prepare_model_riheng(self):
        '''load 网络模型，（更新输出层）
        '''
        sym, arg_params, aux_params = mx.model.load_checkpoint('/workspace/model/resnet-50', 0)
        num_hidden = 10
        (new_sym, new_args) = self.get_fine_tune_model(sym, arg_params, num_hidden)

        # get information of ctx
        gpu_count = self.solver_config.get('gpu_counts', 0)
        ctx = [mx.cpu()] if gpu_count == 0 else [
            mx.gpu(i) for i in xrange(gpu_count)
        ]

        mod = mx.mod.Module(symbol=new_sym, context=ctx)
        mod.bind(data_shapes=self.train_data.provide_data,
                 label_shapes=self.train_data.provide_label)

        logger.info("set pretrained weights")
        mod.set_params(new_args, aux_params, allow_missing=True)
        self.mod = mod

    def start_new_training(self):
        try:
            # 绑定信号，如果是接收到信号，表示用户自己选择退出训练实例
            # 训练实例状态为正常结束
            SUPPORTED_SIGNALS = (signal.SIGINT, signal.SIGTERM,)
            for signum in SUPPORTED_SIGNALS:
                try:
                    signal.signal(signum, self.signal_handler)
                    logger.info("Bind signal '%s' success to %s",
                                signum, self.signal_handler)
                except Exception as identifier:
                    logger.warning(
                        "Bind signal '%s' failed, err: %s", signum, identifier)

            # AVA-SDK 初始化一个训练实例
            self.train_ins = train.TrainInstance()

            logger.info("start new tarining, training_ins_id: %s",
                        self.train_ins.get_training_ins_id())

            logger.info("prepare_train_config")
            self.prepare_train_config()
            logger.info("prepare_solver_config")
            self.prepare_solver_config()
            logger.info("prepare_sampleset_config")
            self.prepare_sampleset_data()
            logger.info("prepare_model")
            #self.prepare_model()
            self.prepare_model_riheng()

            opts = self.train_config
            opts.update(self.solver_config)
            fit_args = {k: opts.get(k) for k in FIT_KWARGS_KEYS}
            logger.info("fit args: %s" % fit_args)
            self.mod.fit(self.train_data,
                        eval_data=self.val_data,
                        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
                        **fit_args)

            logger.info("training finish")
            err_msg = ""
        except Exception as err:
            err_msg = "training failed, err: %s" % (err)
            logger.info(err_msg)
            traceback.print_exc(file=sys.stderr)

        self.clean_up(err_msg=err_msg)

    def clean_up(self, err_msg=""):
        # AVA-SDK 实例结束，需要调用 done，完成状态上报以及清理工作
        if self.train_ins == None:
            return
        self.train_ins.done(err_msg=err_msg)

    def signal_handler(self, signum, stack):
        logger.info("received signal: %s, do clean_up", signum)
        self.clean_up()
        sys.exit()

if __name__ == "__main__":
    worker = MXNetTrainingWorker()
    worker.start_new_training()
