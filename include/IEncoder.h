//
//  IEncoder.h
//  Irrlicht_OSX
//
//  Created by 宋庭聿 on 2021/11/22.
//

#ifndef __I_ENCODER_H_INCLUDED__
#define __I_ENCODER_H_INCLUDED__


extern "C"
{
#include<libavdevice/avdevice.h>
#include<libavutil/frame.h>
#include<libavutil/mem.h>
#include<libavcodec/avcodec.h>
#include<libavformat/avformat.h>
#include<libavutil/opt.h>
#include<libavutil/imgutils.h>
#include<libswscale/swscale.h>
};

#include <iostream>

namespace irr
{
namespace video
{

// TODO: window width and height unknown
class IEncoder{
private:
    int frame_count;
    uint8_t * in_buf[2];
    char* out_filename;
    FILE* fout;
    AVCodec* codec;
    AVCodecContext* codecCtx;
    AVFormatContext* ofctx;
    AVStream* stream;
    AVOutputFormat* outputFormat;
    AVPacket* pkt;
    AVFrame* frameYUV;
    SwsContext* swsContext;
    int SCR_WIDTH;
    int SCR_HEIGHT;
    int inlinesize[8] = {0};
    bool dump_video_option ;
public:
    IEncoder(const core::dimension2d<u32>& size){
        ofctx = nullptr;
        stream = nullptr;
        outputFormat = nullptr;
        codec = nullptr;
        codecCtx = nullptr;
        pkt = nullptr;
        frameYUV = nullptr;
        swsContext = nullptr;
        out_filename = "quake3d.h264";
        frame_count = 0;
        SCR_WIDTH = size.Width;
        SCR_HEIGHT = size.Height;
        inlinesize[0] = SCR_WIDTH*3;
        in_buf[0] = (uint8_t*)malloc(sizeof(uint8_t)*SCR_WIDTH*SCR_HEIGHT*3);
        in_buf[1] = nullptr;
        dump_video_option = false;
    }

    void Init()
    {
        av_register_all();
        avcodec_register_all();
        avformat_network_init();
        int ret = 0;
        in_buf[0] = (uint8_t*)malloc(sizeof(uint8_t)*SCR_HEIGHT*SCR_WIDTH*3);
        in_buf[1] = nullptr;

        // Output file context

        avformat_alloc_output_context2(&ofctx,NULL,NULL,out_filename);
        outputFormat = ofctx->oformat;

        if(avio_open2(&ofctx->pb,out_filename,AVIO_FLAG_WRITE, nullptr,nullptr) < 0)
        {
            std::cout << "open outfile error" << std::endl;
            return;
        }

        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        stream = avformat_new_stream(ofctx,codec);
        codecCtx = avcodec_alloc_context3(codec);

        if(stream == nullptr){
            std::cout<<"Create new stream fails"<<std::endl;
            return;
        }
        const AVRational dst_fps = {25,1};
        codecCtx->codec_tag = 0;
        codecCtx->codec_id = AV_CODEC_ID_H264;
        codecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
        codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
        codecCtx->width = SCR_WIDTH;
        codecCtx->height = SCR_HEIGHT;
        codecCtx->gop_size = 25;
        codecCtx->framerate = dst_fps;
        codecCtx->time_base = av_inv_q(dst_fps);
        //codecCtx->bit_rate = SCR_HEIGHT * SCR_WIDTH * 3;
        codecCtx->max_b_frames = 0;
        if(ofctx->oformat->flags & AVFMT_GLOBALHEADER){
            codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        ret = avcodec_parameters_from_context(stream->codecpar,codecCtx);
        if(ret < 0){
            std::cout<<"Could not initialize stream codec parameters"<<std::endl;
            exit(1);
        }

        AVDictionary* para = nullptr;
        if(codecCtx->codec_id == AV_CODEC_ID_H264){
            av_dict_set(&para,"preset","slow",0);
            av_dict_set(&para,"tune","zerolatency",0);
        }
        ret = avcodec_open2(codecCtx,codec,&para);
        if(ret < 0){
            std::cout<<"avcodec_open2 failed"<<std::endl;
            return;
        }
        stream->codecpar->extradata_size = codecCtx->extradata_size;
        stream->codecpar->extradata = codecCtx->extradata;
        avformat_write_header(ofctx,NULL);
        pkt = av_packet_alloc();
        if(!pkt)
            exit(1);

        frameYUV = av_frame_alloc();
        if(!frameYUV){
            std::cerr<<"Could not allocate video frame\n";
            exit(1);
        }
        frameYUV->format = AV_PIX_FMT_YUV420P;
        frameYUV->width = SCR_WIDTH;
        frameYUV->height = SCR_HEIGHT;
        ret = av_frame_get_buffer(frameYUV,0);
        if (ret < 0) {
            fprintf(stderr, "Could not allocate the video frame data\n");
            exit(1);
        }

        fout = fopen(out_filename,"wb");
        assert(fout!=NULL);
        swsContext = sws_getContext(SCR_WIDTH,SCR_HEIGHT,AV_PIX_FMT_RGB24,
                                    SCR_WIDTH,SCR_HEIGHT,AV_PIX_FMT_YUV420P,
                                    SWS_BICUBIC,NULL,NULL,NULL);
    }

    void GenOnePkt(uint8_t* buffer,uint8_t** ret_buf,int& ret_buf_size)
    {
        // TODO: please reverse the picture upside down
        flip(&buffer);
        memcpy(in_buf[0],buffer,sizeof(uint8_t)*SCR_HEIGHT*SCR_WIDTH*3);
        in_buf[1] = nullptr;
        // Dump
        //rgb24toppm(buffer,SCR_WIDTH,SCR_HEIGHT);
        //rgb24toppm(in_buf[0],SCR_WIDTH,SCR_HEIGHT);
        int height = sws_scale(swsContext,(const uint8_t* const*)in_buf,inlinesize,0,SCR_HEIGHT,
                               frameYUV->data,frameYUV->linesize);
        if(height <= 0) exit(1);
        // TODO: whether pts info needed should be further discuss
        //frameYUV->pts = (frame_count++)*(stream->time_base.den)/((stream->time_base.num)*25);
        frameYUV->pts = AV_NOPTS_VALUE;
        int ret = avcodec_send_frame(codecCtx,frameYUV);
        if(ret < 0){
            printf("Error sending a frame for encoding");
            exit(1);
        }
        while(ret >= 0){
            ret = avcodec_receive_packet(codecCtx,pkt);
            if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF){
                return;
            }
            else if(ret < 0){
                std::cerr<<"error during encoding"<<std::endl;
                exit(1);
            }
            if(dump_video_option){
                DumpLocalVideo();
            }
            ret_buf_size = pkt->size;
            //printf("%d\n",ret_buf_size);
            *ret_buf = (uint8_t*)malloc(ret_buf_size*sizeof(uint8_t));
            memcpy(*ret_buf,pkt->data,sizeof(uint8_t)*ret_buf_size);
            // TODO: send this pkt data to rtsp out file.
            av_packet_unref(pkt);
        }
    }

    void DumpLocalVideo()
    {
        pkt->stream_index = stream->index;
        int er = av_write_frame(ofctx,pkt);
        if(er < 0){
            std::cout<<"write frame error"<<std::endl;
        }
        fwrite(pkt->data,1,pkt->size,fout);
        //count--;
    }

    void FlushEncoder(int streamIndex)
    {
        int ret;
        int got_frame;
        AVPacket enc_pkt;
        if(!(ofctx->streams[0]->codec->codec->capabilities & AV_CODEC_CAP_DELAY))
            return;
        while(1){
            enc_pkt.data = nullptr;
            enc_pkt.size = 0;
            av_init_packet(&enc_pkt);
            ret = avcodec_encode_video2(ofctx->streams[streamIndex]->codec,&enc_pkt, nullptr,&got_frame);
            av_frame_free(NULL);
            if(ret < 0)
                break;
            if(!got_frame){
                ret = 0;
                break;
            }
            printf("Flush Encoder: Succeed to encode 1 frame!\tsize:%5d\n",enc_pkt.size);
            ret = av_write_frame(ofctx,&enc_pkt);
            if(ret < 0)
                break;
        }
        return;
    }

    void EndEncode()
    {
        FlushEncoder(0);
        av_write_trailer(ofctx);
        if(stream){
            avcodec_close(stream->codec);
        }
        avio_close(ofctx->pb);
        av_packet_unref(pkt);
        avformat_free_context(ofctx);
        //avcodec_free_context(&codecCtx);
        av_frame_free(&frameYUV);
        //sav_packet_free(&pkt);
        sws_freeContext(swsContext);
        fclose(fout);
    }
    
private:
    // the buf is up-down reversed
    void rgb24toppm(uint8_t *buf, int width, int height)
    {
        FILE* fp = fopen("rgb24to.ppm","wb+");
        write_ppm_header(fp);
        for(int j = 0; j < height;j++){
            fwrite(buf+j*width*3,1,width*3,fp);
        }
        fclose(fp);
        return;
    }
    void write_ppm_header(FILE *fp)
    {
        fprintf(fp,"P6\n");
        fprintf(fp,"%d %d\n",SCR_WIDTH,SCR_HEIGHT);
        fprintf(fp,"%d\n",255);
    }

    void flip(uint8_t** buf)
    {
        int totalLength = SCR_HEIGHT*SCR_WIDTH*3;
        int oneLineLength = SCR_WIDTH*3;
        uint8_t* tmp = (uint8_t*)malloc(SCR_HEIGHT*SCR_WIDTH*3);
        memcpy(tmp,*buf,SCR_WIDTH*SCR_HEIGHT*3);
        memset(*buf,0,sizeof(uint8_t)*SCR_HEIGHT*SCR_WIDTH*3);
        for(int i = 0; i < SCR_HEIGHT;i++){
            memcpy(*buf+oneLineLength*i,tmp+totalLength-oneLineLength*(i+1),oneLineLength);
        }
        // rgb24toppm(*buf,SCR_WIDTH,SCR_HEIGHT);
    }};

} // end namespace video
} // end namespace irr

#endif /* IEncoder_h */
