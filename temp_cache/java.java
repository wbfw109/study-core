import java.util.*;
public class java {
      public static List<List<Integer>> threeSum(int[] nums) {      
          return nums;
      }
  
      public static void main(String... args){
          List<List<Integer>> triplets = threeSum(new int[]{2,0,-1,1,-2,3,3});
          for(int i=0;i<triplets.size();++i) System.out.println(triplets.get(i).toString());
      }
  
  }
}
