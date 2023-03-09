import java.util.Arrays;

public class Hello {
    public static void main(String[] args){
        System.out.println("Hello world");

        int[] student_no = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        int[] checked_index = {10, 3, 5, 6, 7, 8, 9, 2};
        int[] result = new int[2];
        int next_index = 0;

        for(int i=0; i< student_no.length; i++){
            boolean is_found = false;
            for(int j=0; j< checked_index.length; j++){    
                if(student_no[i] == checked_index[j]){
                    is_found = true;
                    break;
                }
            }
            
            if (!is_found) {
                // System.out.println(i);
                result[next_index] = student_no[i];
                next_index+=1;
                if (next_index>1){
                    break;
                }
            }
        }
        System.out.println(Arrays.toString(result));


    }
}