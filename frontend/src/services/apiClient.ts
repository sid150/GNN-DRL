import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import { config } from '@/config';
import type { APIResponse } from '@/types';

class APIClient {
  private client: AxiosInstance;
  private authToken: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: config.apiBaseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401 && config.enableAuth) {
          // Handle authentication error
          this.authToken = null;
          // Could dispatch logout action here
        }
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: AxiosError): Error {
    if (error.response) {
      const message = (error.response.data as any)?.error || error.message;
      return new Error(`API Error: ${message}`);
    } else if (error.request) {
      return new Error('Network Error: No response received from server');
    } else {
      return new Error(`Request Error: ${error.message}`);
    }
  }

  setAuthToken(token: string) {
    this.authToken = token;
  }

  clearAuthToken() {
    this.authToken = null;
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<APIResponse<T>> {
    const response = await this.client.get(url, config);
    // Backend returns data directly, wrap it in APIResponse format
    return {
      success: true,
      data: response.data,
      timestamp: new Date().toISOString()
    };
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<APIResponse<T>> {
    const response = await this.client.post(url, data, config);
    return {
      success: true,
      data: response.data,
      timestamp: new Date().toISOString()
    };
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<APIResponse<T>> {
    const response = await this.client.put(url, data, config);
    return {
      success: true,
      data: response.data,
      timestamp: new Date().toISOString()
    };
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<APIResponse<T>> {
    const response = await this.client.delete(url, config);
    return {
      success: true,
      data: response.data,
      timestamp: new Date().toISOString()
    };
  }

  async download(url: string, filename: string): Promise<void> {
    const response = await this.client.get(url, {
      responseType: 'blob',
    });
    
    const blob = new Blob([response.data]);
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
  }

  async upload(url: string, file: File, onProgress?: (progress: number) => void): Promise<APIResponse<any>> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });

    return response.data;
  }
}

export const apiClient = new APIClient();
