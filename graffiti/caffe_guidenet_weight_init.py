# coding=utf-8
import caffe

net = caffe.Net("/home/wangbin/github/RFCN-FasterRCNN/objectDetection/UISEE-FRCNN-3/model_config/train.prototxt",
                "/media/wangbin/8057840b-9a1e-48c9-aa84-d353a6ba1090/UISEE/"
                "caffe_models/PVANET/PVANET-LITE/PVANET-LITE.caffemodel", caffe.TEST)\

print(type(net.params))