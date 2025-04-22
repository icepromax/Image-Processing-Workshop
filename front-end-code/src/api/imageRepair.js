import axios from 'axios';


/**
 * 图像修复API
 * @param {File|Blob} imageFile - 要修复的图片文件
 * @param {string} model - 选择的模型ID
 * @returns {Promise<string>} - 返回修复后的图片Base64数据
 */
export const repairImage = async (imageFile, model) => {
    try {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('model', model);
        console.log(model)
        const response = await axios.post(`/repair`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            },
            timeout: 120000
        });

        // 处理返回的图片URL
        const imageUrl = response.data.repairedImage;

        // 如果返回的是相对路径，加上后端基础URL
        if (imageUrl.startsWith('/')) {
            return ` ${axios.defaults.baseURL}${imageUrl}`;
        }

        return imageUrl;
    } catch (error) {
        console.error('API调用失败:', error);
        throw new Error('图片修复失败，请重试');
    }
};

/**
 * 获取示例图片列表
 * @returns {Promise<Array>}
 */
// imageRepair.js

import sample1Thumb from '@/assets/images/cyclegan样例.png'
import sample1Full from '@/assets/images/cyclegan样例.png'
import sample2Thumb from '@/assets/images/图像高清化样例.png'
import sample2Full from '@/assets/images/图像高清化样例.png'
import sample3Thumb from '@/assets/images/图像高清化样例2.png'
import sample3Full from '@/assets/images/图像高清化样例2.png'
import sample4Thumb from '@/assets/images/图像高清化样例3.jpg'
import sample4Full from '@/assets/images/图像高清化样例3.jpg'
import sample5Thumb from '@/assets/images/图像修复样例1.png'
import sample5Full from '@/assets/images/图像修复样例1.png'

export const getSampleImages = async () => {
    return [
        {
            id: 'cyclegan',
            name: '风格迁移样例1',
            thumb: sample1Thumb,
            full: sample1Full
        },
        {
            id: 'real-esrgan',
            name: '图像高清化样例1',
            thumb: sample2Thumb,
            full: sample2Full
        },
        {
            id: 'real-esrgan',
            name: '图像高清化样例2',
            thumb: sample3Thumb,
            full: sample3Full
        },
        {
            id: 'real-esrgan',
            name: '图像高清化样例3',
            thumb: sample4Thumb,
            full: sample4Full
        },
        {
            id: 'Gan_unet',
            name: '图像修复样例1',
            thumb: sample5Thumb,
            full: sample5Full
        },

    ];
};

/**
 * 获取可用模型列表
 * @returns {Promise<Array>}
 */
export const getAvailableModels = async () => {
    // 这里可以是实际API调用，或返回本地数据
    return [
        { id: 'cyclegan', name: 'CycleGan', tag: '莫奈风格迁移', tagColor: '#3498db'},
        { id: 'real-esrgan', name: 'Real-ESRGAN', tag: '图像高清化', tagColor: '#2ecc71' },
        { id: 'Gan_unet', name: 'Gan_unet', tag: '图像修复', tagColor: '#e74c3c' }
    ];
};
