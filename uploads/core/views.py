from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
from uploads import shared_values as shared
from uploads.dataset.dataset import MURA_Dataset
from torch.utils.data import DataLoader
#from tqdm import tqdm
from torch.autograd import Variable
import torch as t
#import csv
import cv2
import numpy as np
import os
from django.template import loader
#from django.http import HttpResponse

def home(request):
    documents = Document.objects.all()
    #documents = Document.objects.all()
    if request.method == 'POST' and request.FILES['myfile']:
        print("upload")
        model = shared.model_arr[0]# densenet69
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        prob, cam_output_file = mura_inference(model,uploaded_file_url )
        result = "{0}".format("판독결과 골격계 비정상 소견 입니다. (Positive)" if prob==1 else "판독결과 골격계 정상 소견 입니다. (Negative)")
        #
        template = loader.get_template('core/simple_upload.html')
        # context = {
        #     'uploaded_file_url': cam_output_file,
        #     'result': result
        # }
        # response = HttpResponse(template.render(context, request))
        #result = kwargs['arg1'] + kwargs['arg2']
        # kwargs['result'] = result
        # kwargs['uploaded_file_url'] = cam_output_file
        #return render(request, 'core/simple_upload.html', kwargs)
        #return response
        return render(request, 'core/home.html', {
            'uploaded_file_url': cam_output_file,
            'result' : result
        })
        # return redirect('simple_upload')
        # # return render(request, 'core/image_result_box.html', {
        # #     'uploaded_file_url': cam_output_file,
        # #     'result': result
        # # })


    #return render(request, 'core/simple_upload.html')
    return render(request, 'core/home.html', { 'documents': documents })
    #return render(request, 'core/simple_upload.html')


def simple_upload(request):
    #documents = Document.objects.all()
    if request.method == 'POST' and request.FILES['myfile']:
        print("upload")
        model = shared.model_arr[0]# densenet69
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        prob, cam_output_file = mura_inference(model,uploaded_file_url )
        result = "{0}".format("판독결과 골격계 비정상 소견 입니다. (Positive)" if prob==1 else "판독결과 골격계 정상 소견 입니다. (Negative)")
        #
        template = loader.get_template('core/simple_upload.html')
        # context = {
        #     'uploaded_file_url': cam_output_file,
        #     'result': result
        # }
        # response = HttpResponse(template.render(context, request))
        #result = kwargs['arg1'] + kwargs['arg2']
        # kwargs['result'] = result
        # kwargs['uploaded_file_url'] = cam_output_file
        #return render(request, 'core/simple_upload.html', kwargs)
        #return response
        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': cam_output_file,
            'result' : result
        })
        # return redirect('simple_upload')
        # # return render(request, 'core/image_result_box.html', {
        # #     'uploaded_file_url': cam_output_file,
        # #     'result': result
        # # })


    return render(request, 'core/simple_upload.html')


def mura_inference(model, uploaded_file_url):
    model.eval()
    data_root = '/Users/yewoo/dev/simple-file-upload'

    # data
    test_data = MURA_Dataset(data_root, uploaded_file_url, 'file',  train=False, test=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    (data, label, path, body_part)  = list(test_dataloader)[0]

    input = Variable(data, volatile=True)

    cam_output_file, idx, probs = mura_cam(model, input, uploaded_file_url )
    return idx, cam_output_file


def mura_cam(model, input_data, uploaded_file_url ):
    # hook the feature extractor
    features_blobs = []
    finalconv_name = 'features'

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    def returnCAM2(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (320, 320)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    score = model(input_data)

    # print(outputs)
    h_x = t.nn.functional.softmax(score, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    CAMs = returnCAM2(features_blobs[0], weight_softmax, [idx[0]])

    img = cv2.imread("."+uploaded_file_url)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = (heatmap * 0.3 + img * 0.5)

    res = np.hstack((img, result))

    base = os.path.basename(uploaded_file_url)
    filename = os.path.splitext(base)[0]
    base_dir = os.path.dirname(uploaded_file_url)

    cam_output_file = base_dir + "/" + filename + "_cam.png"
    cv2.imwrite("." + cam_output_file, res)
    return cam_output_file, idx[0], probs[0]


# def write_csv(results, file_name):
#     with open(file_name, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['image', 'probability'])
#         writer.writerows(results)

def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
